# coding: utf-8
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import json
import os
import os.path as osp
import time
from glob import glob

import mmcv
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env
from PIL import Image
from tqdm import tqdm

from utils import get_area, setup_logger

parser = argparse.ArgumentParser(description="Train a vehicle detector")
parser.add_argument(
    "--config",
    help="train config file path",
    default="./configs/lane_detection_config.py",
)
parser.add_argument(
    "--no-validate",
    action="store_true",
    help="whether not to evaluate the checkpoint during training",
)
group_gpus = parser.add_mutually_exclusive_group()
group_gpus.add_argument(
    "--gpus",
    type=int,
    help="number of gpus to use " "(only applicable to non-distributed training)",
)
group_gpus.add_argument(
    "--gpu-ids",
    type=int,
    nargs="+",
    help="ids of gpus to use " "(only applicable to non-distributed training)",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="whether to set deterministic options for CUDNN backend.",
)
parser.add_argument(
    "--launcher",
    choices=["none", "pytorch", "slurm", "mpi"],
    default="none",
    help="job launcher",
)
parser.add_argument(
    "--no-merging",
    action="store_true",
    help="whether to merge annotations before training. "
    "Set true when there exists merged json file",
)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def merge_annotation(annot_list, save_path):
    images = []
    annotations = []
    annot_id = 1
    image_id = 0
    for annot in tqdm(annot_list):
        with open(annot, "r") as f:
            json_data = json.load(f)
        object_data = json_data["data_set_info"]["data"]
        img_path = annot.replace("ANNOTATION", "IMAGE").replace("json", "jpg")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)

        img_info = {}
        img_info["id"] = image_id
        file_name = osp.basename(img_path)
        img_info["file_name"] = copy.deepcopy(file_name)
        img_info["height"] = copy.deepcopy(img.size[1])
        img_info["width"] = copy.deepcopy(img.size[0])
        images.append(copy.deepcopy(img_info))
        img.close()

        # annotation은 segmentation / iscrowd, image_id, \
        # category_id, id, bbox, area
        for target in object_data:
            obj_label = target["value"]["object_Label"]
            if "lane_type" in obj_label.keys():
                category = obj_label["lane_type"]
            else:
                continue
            ann = {}
            if category in [
                "lane_blue",
                "lane_shoulder",
                "lane_white",
                "lane_yellow",
            ]:
                points = target["value"]["points"]
                temp_points = copy.deepcopy(points)
                # deepcopy를 해야 둘다 변경되지 않음
                area = get_area(temp_points)
                if category == "lane_blue":
                    ann["category_id"] = 1
                elif category == "lane_shoulder":
                    ann["category_id"] = 2
                elif category == "lane_white":
                    ann["category_id"] = 3
                elif category == "lane_yellow":
                    ann["category_id"] = 4
                segmentation = []
                seg_x = []
                seg_y = []
                for point in points:
                    segmentation.append(point["x"])
                    seg_x.append(point["x"])
                    segmentation.append(point["y"])
                    seg_y.append(point["y"])
                bbox = [
                    min(seg_x),
                    min(seg_y),
                    max(seg_x) - min(seg_x),
                    max(seg_y) - min(seg_y),
                ]
                ann["bbox"] = bbox
                ann["segmentation"] = [segmentation]
                ann["area"] = area
                ann["image_id"] = image_id
                ann["iscrowd"] = 0
                ann["id"] = annot_id
                annot_id += 1
            if ann:
                annotations.append(ann)
        image_id += 1
    merged_data = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"supercategory": "lane", "id": 1, "name": "lane_blue"},
            {"supercategory": "lane", "id": 2, "name": "lane_shoulder"},
            {"supercategory": "lane", "id": 3, "name": "lane_white"},
            {"supercategory": "lane", "id": 4, "name": "lane_yellow"},
        ],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent="\t")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Set configs
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = setup_logger(log_file, name=__name__)
    logger.info("Start training!!")
    logger.info(f"Read config from {args.config}")

    # build coco-style dataset
    if not args.no_merging:
        for split in ["train", "valid"]:
            annot_list = glob(osp.join(cfg.data_source, split, "ANNOTATION/*.json"))
            logger.info(f"Merging {len(annot_list)} files in {split} dataset.")
            if not osp.exists(cfg.ann_source):
                os.makedirs(cfg.ann_source)
            save_path = osp.join(cfg.ann_source, f"{split}.json")
            merge_annotation(annot_list, save_path)
    for _, _, filename in os.walk(osp.join(cfg.data_source, "train/IMAGE")):
        if len(filename) > 0:
            logger.info(f"Num images => {filename}")
            logger.info(f"File list => {filename}")

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, " f"deterministic: {args.deterministic}"
        )
        set_random_seed(cfg.seed, deterministic=args.deterministic)
    meta["seed"] = cfg.seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES
        )
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
