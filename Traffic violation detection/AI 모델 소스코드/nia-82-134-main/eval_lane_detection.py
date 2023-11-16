#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import logging
import os
import os.path as osp
import time
import warnings
from glob import glob
from statistics import mean

import mmcv
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import auc  # Area Under Curve

from utils import merge_lane_annotation, setup_logger

warnings.filterwarnings(action="ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="config file for MMdetection test ",
    default="./configs/lane_detection_config.py",
)
parser.add_argument(
    "--launcher",
    choices=["none", "pytorch", "slurm", "mpi"],
    default="none",
    help="job launcher",
)
parser.add_argument("--save_dir", default="./eval_logs/lane_detection")
parser.add_argument("--ckpt", default="./best_models/lane_detection_model.pth")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--test_batch_size", type=int, default=16)
parser.add_argument("--no-merging", action="store_true")
args = parser.parse_args()
logger = setup_logger(
    os.path.join(args.save_dir, "eval_log.txt"),
    name=__name__,
    mode="w",
    level=logging.DEBUG,
)


def parse_args():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def get_test_result(cfg):
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = args.test_batch_size
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = args.test_batch_size
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.save_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.save_dir))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        osp.join(args.save_dir, f"eval_{timestamp}.json")

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.test_batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.ckpt, map_location="cpu")

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, False, None, 0.3)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, "/tmp", True)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {"jsonfile_prefix": osp.join(args.save_dir, "result")}
        dataset.format_results(outputs, **kwargs)


def main():
    logger.info("Start running eval_lane_detection.py")
    start_time = time.time()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if not args.no_merging:
        for split in ["test"]:
            annot_list = glob(osp.join(cfg.data_source, split, "ANNOTATION/*.json"))
            logger.info(f"Merging {len(annot_list)} files in {split} dataset.")
            if not osp.exists(cfg.ann_source):
                os.makedirs(cfg.ann_source)
            save_path = osp.join(cfg.ann_source, f"{split}.json")
            merge_lane_annotation(annot_list, save_path)
    cfg.data.test.test_mode = True
    for _, _, filename in os.walk(os.path.join(cfg.data_source, "test/IMAGE")):
        if len(filename) > 0:
            logger.info(f"Num images => {len(filename)}")
            logger.info(f"File list => {filename}")

    get_test_result(cfg)

    res_file = osp.join(args.save_dir, "result.segm.json")
    gt_ann_file = cfg.data.test.ann_file
    cocoGt = COCO(gt_ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()

    # Build dataset
    dataset = build_dataset(cfg.data.test)
    logger.debug(dataset)

    # Get Prediction result
    res_df = []
    for elem in cocoEval.evalImgs:
        if elem is None:
            continue
        if elem["aRng"] != [0, 10000000000.0]:
            continue
        if len(elem["dtScores"]) == 1 and elem["dtScores"][0] < 0.5:
            continue
        if len(elem["gtIds"]) == 0:
            continue

        # add a row in df
        row_dict = {
            "Data ID": elem["image_id"],
            "Class - Ground Truth": elem["category_id"],
        }
        for i, gt_id in enumerate(elem["gtIds"]):
            pred_cat = elem["category_id"]
            max_score = 0
            # no matching dts or have wrong category
            if len(elem["dtIds"]) == 0:
                preds = cocoEval.cocoDt.imgToAnns[elem["image_id"]]
                pred_cat = -1
                for p in preds:
                    if p["score"] > max_score:
                        pred_cat = p["category_id"]
                        max_score = p["score"]
            # more than one matching dt
            else:
                dtMatches = elem["dtMatches"][0]
                match_idx = np.where(dtMatches == gt_id)[0]
                # no matching dt
                if len(match_idx) == 0:
                    pred_cat = -1
                # there is a matching dt
                else:
                    max_score = elem["dtScores"][match_idx[0]]
            row_dict.update(
                {"Class - Predict": pred_cat, "Confidence level": max_score}
            )
            res_df.append(row_dict)
    res_df = pd.DataFrame(res_df)

    # Get Confusion matrix, precision and recall
    df_list = []
    for label in np.unique(res_df["Class - Ground Truth"]):
        if label < 0:
            continue
        curr_df = copy.deepcopy(
            res_df.loc[
                res_df["Class - Ground Truth"] == label,
            ]
        )
        curr_df.sort_values("Confidence level", inplace=True, ascending=False)
        curr_df.reset_index(drop=True, inplace=True)
        fn = sum(curr_df["Class - Predict"] == label)
        add_dict = {
            "Confusion matrix": [],
            "누적 TP": [],
            "누적 FP": [],
            "Precision": [],
            "Recall": [],
        }
        agg_tp = 0
        agg_fp = 0
        for i, row in curr_df.iterrows():
            if row["Class - Ground Truth"] == row["Class - Predict"]:
                agg_tp += 1
                fn -= 1
            else:
                agg_fp += 1
            add_dict["Confusion matrix"].append(f"{agg_tp},0,{agg_fp},{fn}")
            add_dict["누적 TP"].append(agg_tp)
            add_dict["누적 FP"].append(agg_fp)
            add_dict["Precision"].append(agg_tp / (agg_tp + agg_fp))
            add_dict["Recall"].append(agg_tp / (agg_tp + fn))
        tmp_df = pd.DataFrame(add_dict)
        curr_df = pd.concat([curr_df, tmp_df], axis=1)
        df_list.append(curr_df)
    agg_df = pd.concat(df_list, axis=0)
    agg_df.reset_index(drop=True, inplace=True)

    ap_list = []
    for i, c_name in enumerate(cfg.classes):
        curr_recall = agg_df.loc[agg_df["Class - Ground Truth"] == i + 1, "Recall"]
        curr_precision = agg_df.loc[
            agg_df["Class - Ground Truth"] == i + 1, "Precision"
        ]
        auc_result = auc(curr_recall, curr_precision)
        logger.info(f"AP of {c_name} => {auc_result}")
        ap_list.append(auc_result)

        plt.plot(curr_recall, curr_precision)
        plt.xlim([0, 1])  # X축의 범위: [xmin, xmax]
        plt.ylim([0, 1])  # Y축의 범위: [ymin, ymax]
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.fill_between(curr_recall[0:-1], curr_precision[0:-1], alpha=0.5)
        plot_title = c_name + "_precision-recall_curve"
        plt.title(plot_title)

        font = {
            "family": "Times New Roman",
            "color": "black",
            "weight": "bold",
            "size": 12,
            "alpha": 0.7,
        }

        plt.text(0.75, 0.95, "AP = {}".format(round(auc_result, 3)), fontdict=font)
        plt.savefig(
            os.path.join(args.save_dir, plot_title + ".png"),
            dpi=200,
            facecolor="#eeeeee",
            edgecolor="black",
        )
        plt.clf()

    # mAP 계산
    eval_mAP = mean(ap_list)
    logger.info(f"eval_mAp => {eval_mAP}")

    # 빈 열로 간격 추가
    agg_df[""] = None
    # AP에 대한 DataFrame 생성
    ap_df_dict = {"Class ID": [], "AP": [], "mAP": [eval_mAP]}
    for i, ap in enumerate(ap_list):
        ap_df_dict["Class ID"].append(int(i + 1))
        ap_df_dict["AP"].append(ap)
    ap_df = pd.DataFrame.from_dict(ap_df_dict, orient="index")

    # AP dataframe과 결합
    # 크기가 맞지 않아서 모자란 부분은 자동적으로 NaN 값으로 채워짐.
    result_df = pd.concat([agg_df, ap_df.transpose()], axis=1)

    # AP, mAP에 있는 NaN값 없애기
    result_df.reset_index(drop=True, inplace=True)
    result_df.to_csv(
        os.path.join(args.save_dir, "eval_lane_merged_df.csv"),
        encoding="utf-8",
        index_label="No.",
    )
    logger.info(f"Evaluation takes {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
