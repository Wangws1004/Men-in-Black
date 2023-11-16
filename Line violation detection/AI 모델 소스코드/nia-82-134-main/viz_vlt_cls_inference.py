# coding: utf-8
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import datasets, transforms

from utils import setup_logger
from vlt_cls.model import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default="configs/vlt_cls_model_config.yaml")
args = parser.parse_args()
logger = setup_logger(name=__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./viz_output/vlt_cls"


def main():
    logger.info("Start visualization!!")
    logger.info(f"Read config from {args.config_file}")
    cfg = OmegaConf.load(args.config_file)
    if cfg.load_model is None or not os.path.exists(cfg.load_model):
        logger.error("Model file does not exist or cfg.load_model is None.")
        return
    logger.info(f"=========configs========\n")
    logger.info(OmegaConf.to_yaml(cfg))
    # set save path
    model_name = os.path.splitext(os.path.basename(cfg.load_model))[0]
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, f"{model_name}_viz.png")

    # get image loader
    data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.0181, 0.0304, 0.0147], [0.1199, 0.1648, 0.1065]),
        ]
    )
    image_dataset = datasets.ImageFolder(
        os.path.join(cfg.data_dir, "test"), transform=data_transform
    )
    data_loader = torch.utils.data.DataLoader(
        image_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4
    )
    class_names = image_dataset.classes

    # get model
    clf_model = get_model(len(class_names), cfg.load_model, device)
    clf_model.eval()

    images_so_far = 0
    num_images = 6
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, preds = torch.max(clf_model(inputs), 1)
            fig = plt.figure()
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"Predicted: {class_names[preds[j]]}")
                inp = inputs.cpu().data[j]
                inp = inp.numpy().transpose((1, 2, 0))
                mean = np.array([0.0181, 0.0304, 0.0147])
                std = np.array([0.1199, 0.1648, 0.1065])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                ax.imshow(inp)
                if images_so_far == num_images:
                    fig.savefig(save_path)
                    logger.info("Visualization DONE!!")
                    return


if __name__ == "__main__":
    main()
