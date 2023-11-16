# coding: utf-8
import torch
import torch.nn as nn
from torchvision import models

from utils import setup_logger

logger = setup_logger(name=__name__)


def get_model(num_classes, load_from=None, device="cpu"):
    clf_model = models.resnet18(pretrained=True)
    emb_features = clf_model.fc.in_features
    clf_model.fc = nn.Linear(emb_features, num_classes)
    if load_from is not None and load_from != "":
        logger.info(f"Loading model from {load_from}...")
        loaded_dict = torch.load(load_from, map_location=device)
        clf_model.load_state_dict(loaded_dict["state_dict"])
    clf_model = clf_model.to(device)

    return clf_model
