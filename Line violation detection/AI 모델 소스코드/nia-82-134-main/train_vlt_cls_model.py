# coding: utf-8
import argparse
import copy
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from utils import setup_logger
from vlt_cls.model import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default="./configs/vlt_cls_model_config.yaml")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg = OmegaConf.load(args.config_file)
    log_dir = cfg.model_save_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "train_log.log")
    logger = setup_logger(log_file, name=__name__)
    logger.info("Start training!!")
    logger.info(f"Read config from {args.config_file}")
    logger.info(f"=========configs========\n")
    logger.info(OmegaConf.to_yaml(cfg))

    # data loader
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.0181, 0.0304, 0.0147], [0.1199, 0.1648, 0.1065]
                ),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.0181, 0.0304, 0.0147], [0.1199, 0.1648, 0.1065]
                ),
            ]
        ),
    }
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(cfg.data_dir, x), transform=data_transforms[x]
        )
        for x in ["train", "valid"]
    }
    data_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=cfg.batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "valid"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}
    class_names = image_datasets["train"].classes
    logger.info("Length of loaded datasets:")
    logger.info(f"Train: {dataset_sizes['train']}, " f"Valid: {dataset_sizes['valid']}")
    logger.info(f"Class names => {class_names}")
    for _, dirname, filenames in os.walk(os.path.join(cfg.data_dir, "train")):
        if len(filenames) > 0:
            logger.info(f"length of filenames => {len(filenames)}")
            logger.info(f"files => {filenames}")

    # Start training
    logger.info("Initialize model...")
    clf_model = get_model(len(class_names), cfg.load_model, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf_model.parameters(), lr=cfg.lr, momentum=0.9)
    exp_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma, verbose=True
    )
    # exp_scheduler = lr_scheduler.ExponentialLR(
    #     optimizer, gamma=cfg.lr_gamma, verbose=True
    # )

    logger.info("Start training...")
    start_time = time.time()
    best_model_weights = copy.deepcopy(clf_model.state_dict())
    best_acc = 0.0

    for epoch in range(cfg.num_epochs):
        logger.info("=" * 10)
        logger.info(f"Epoch {epoch}/{cfg.num_epochs - 1}")
        for phase in ["train", "valid"]:
            is_train = phase == "train"
            if is_train:
                clf_model.train()
            else:
                clf_model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    outputs = clf_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if is_train:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # lr scheduler step
            if is_train:
                exp_scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            logger.info(f"{phase} => Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if not is_train and epoch_acc > best_acc:
                logger.info(
                    f"The best acc is improved from "
                    f"{best_acc:.4f} to {epoch_acc:.4f}"
                )
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(clf_model.state_dict())
    time_elapsed = time.time() - start_time
    logger.info(
        f"Training finished in {(time_elapsed // 60):.0f}m "
        f"{(time_elapsed % 60):.0f}"
    )
    logger.info(f"Best val acc: {best_acc:.4f}")
    now = datetime.now().strftime("%y%m%d")
    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)
    model_save_path = os.path.join(cfg.model_save_dir, f"best-{now}.pth")
    torch.save(
        {"config": OmegaConf.to_container(cfg), "state_dict": best_model_weights},
        model_save_path,
    )
    logger.info(f"Best model is saved to {model_save_path}")


if __name__ == "__main__":
    main()
