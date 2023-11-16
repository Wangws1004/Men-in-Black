# coding: utf-8
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import torch
from mmdet.apis import inference_detector, init_detector

from utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./configs/vehicle_detection_config.py")
parser.add_argument("--ckpt", default="./best_models/vehicle_detection_model.pth")
parser.add_argument(
    "--input_image", default="./sample_images/[BLUE]07909B_134242_003.jpg"
)
parser.add_argument("--save_dir", default="./viz_output/vehicle_detection")
args = parser.parse_args()

logger = setup_logger(name=__name__)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():
    logger.info("Start vehicle detection inference")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger.info(f"Outputs will be saved at {args.save_dir}")

    logger.info(f"Input image => {args.input_image}")
    vehicle_model = init_detector(args.config, args.ckpt, device=device)
    bbox_result, segm_result = inference_detector(vehicle_model, args.input_image)
    result_img = vehicle_model.show_result(args.input_image, (bbox_result, segm_result))
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    plt.imsave(
        os.path.join(args.save_dir, os.path.basename(args.input_image)), result_img
    )
    logger.info("Visualize DONE!")


if __name__ == "__main__":
    main()
