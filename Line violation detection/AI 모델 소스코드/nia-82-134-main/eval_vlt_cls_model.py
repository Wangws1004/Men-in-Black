# coding: utf-8
import argparse
import logging
import os
import time

import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torchvision import transforms
from tqdm import tqdm

from utils import setup_logger
from vlt_cls.dataset import VltDataset
from vlt_cls.model import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default="configs/vlt_cls_model_config.yaml")
args = parser.parse_args()
SAVE_DIR = "./eval_logs/vlt_cls_eval_log"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
logger = setup_logger(
    log_file_path=os.path.join(SAVE_DIR, "vlt_clf_eval_log.txt"),
    name=__name__,
    level=logging.INFO,
    mode="w",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    logger.info("Start running eval_violation_classification.py")
    logger.info(f"Read config from {args.config_file}")
    cfg = OmegaConf.load(args.config_file)
    logger.info(f"=========configs========\n")
    logger.info(OmegaConf.to_yaml(cfg))
    start_time = time.time()

    # get image loader
    data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.0181, 0.0304, 0.0147], [0.1199, 0.1648, 0.1065]),
        ]
    )
    image_dataset = VltDataset(
        os.path.join(cfg.data_dir, "test"), transform=data_transform
    )
    data_loader = torch.utils.data.DataLoader(
        image_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4
    )
    class_names = image_dataset.classes
    for _, _, filename in os.walk(os.path.join(cfg.data_dir, "test")):
        if len(filename) > 0:
            logger.info(f"Num images in curr dir => {len(filename)}")
            logger.info(f"File list => {filename}")

    # get model
    clf_model = get_model(len(class_names), cfg.load_model, device)
    clf_model.eval()

    result_list = []
    with torch.no_grad():
        for _, (image_ids, inputs, labels) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, preds = torch.max(clf_model(inputs), 1)
            curr_df = pd.DataFrame(
                {"Class ID": image_ids, "GT": labels.cpu(), "PD": preds.cpu()}
            )
            result_list.append(curr_df)
    result_df = pd.concat(result_list)

    # calculate metrics
    acc_fn = len(result_df)
    acc_tp = 0
    acc_fp = 0

    result_list = []
    result_df.sort_values(by="GT", inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    for i, rows in result_df.iterrows():
        curr_dict = {"Class ID": rows["Class ID"], "GT": rows.GT, "PD": rows.PD}
        if rows.GT == rows.PD:
            acc_tp += 1
            acc_fn -= 1
        else:
            acc_fp += 1

        precision = acc_tp / (acc_tp + acc_fp)
        recall = acc_tp / (acc_tp + acc_fn)
        curr_dict.update(
            {
                "TP": acc_tp,
                "FP": acc_fp,
                "FN": acc_fn,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": (2 * precision * recall) / (precision + recall + 1e-7),
            }
        )
        result_list.append(curr_dict)
    result_df = pd.DataFrame(result_list)
    logger.info(
        f"f1-scores per class => {f1_score(result_df.GT, result_df.PD, average=None)}"
    )
    logger.debug(f"micro => {f1_score(result_df.GT, result_df.PD, average='micro')}")
    logger.debug(f"macro => {f1_score(result_df.GT, result_df.PD, average='macro')}")

    # save evaluation result
    f1_list = []
    per_class_f1 = f1_score(result_df.GT, result_df.PD, average=None)
    for i, c_name in enumerate(class_names):
        f1_list.append({"Class ID": c_name, "F1-Score": per_class_f1[i]})
    f1_df = pd.DataFrame(f1_list)
    avg_f1 = f1_score(result_df.GT, result_df.PD, average="micro")
    f1_df = f1_df.append({"Class ID": "Average", "F1-Score": avg_f1}, ignore_index=True)

    result_df[""] = None
    result_df = pd.concat([result_df, f1_df], axis=1)
    result_df.to_csv(os.path.join(SAVE_DIR, "eval_log.csv"), index=False)

    logger.info(f"Evaluation takes {time.time() - start_time:.1f}s")
    logger.info("DONE Evaluation")


if __name__ == "__main__":
    main()
