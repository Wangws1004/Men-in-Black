# coding: utf-8
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ObjectInfo:
    obj_id: int
    obj_type: str
    category: str
    bbox: np.ndarray
    poly: np.ndarray
    segm: np.ndarray
    score: float
    label: str = "normal"


@dataclass
class ImageInfo:
    image_id: str
    objects: List[ObjectInfo] = None


LANE_LABEL_MAP = ["BG", "white", "blue", "yellow", "shoulder"]


VEHICLE_COLOR_MAP = {
    "vehicle_car": (0, 255, 0),
    "vehicle_bus": (0, 0, 255),
    "vehicle_truck": (0, 255, 0),
    "vehicle_bike": (0, 123, 123),
}

LANE_COLOR_MAP = {
    "lane_white": {
        "single_dashed": (255, 255, 255),
        "left_dashed_double": (255, 255, 255),
        "right_dashed_double": (255, 255, 255),
        "single_solid": (255, 0, 0),
        "double_solid": (255, 0, 0),
    },
    "lane_blue": {
        "single_dashed": (0, 0, 255),
        "left_dashed_double": (0, 0, 255),
        "right_dashed_double": (0, 0, 255),
        "single_solid": (255, 0, 0),
        "double_solid": (255, 0, 0),
    },
    "lane_yellow": {
        "single_dashed": (255, 255, 255),
        "left_dashed_double": (255, 255, 255),
        "right_dashed_double": (255, 255, 255),
        "single_solid": (255, 0, 0),
        "double_solid": (255, 0, 0),
    },
    "lane_shoulder": {
        "single_dashed": (255, 255, 255),
        "left_dashed_double": (255, 253, 51),
        "right_dashed_double": (255, 253, 51),
        "single_solid": (255, 253, 51),
        "double_solid": (255, 253, 51),
    },
}

DANGER_COLOR = (255, 0, 255)
VLT_COLOR = (127, 0, 255)
NORMAL_COLOR = (153, 255, 255)
NEW_SIZE = (320, 280)

VEHICLE_LIST = ["vehicle_car", "vehicle_truck", "vehicle_bus", "vehicle_bike"]

LANE_LABEL_MAP_PREV = [
    "BG",
    "white_single_solid",
    "white_single_dashed",
    "white_double_solid",
    # "white_left_dashed_double",
    # "white_right_dashed_double",
    "blue_single_solid",
    "blue_single_dashed",
    "blue_double_solid",
    # "blue_left_dashed_double",
    # "blue_left_dashed_double",
    "yellow_single_solid",
    # "yellow_single_dashed",
    "yellow_double_solid",
    # "yellow_left_dashed_double",
    # "yellow_left_dashed_double",
    "shoulder_single_solid",
    "shoulder_single_dashed",
    "shoulder_double_solid",
    # "shoulder_left_dashed_double",
    # "shoulder_left_dashed_double",
    "other",
]

VIOLATION_MAP = {
    "vehicle_car": [
        "white_single_solid",
        "white_double_solid",
        "blue_single_solid",
        "blue_double_solid",
        "blue_single_dashed",
        "blue_left_dashed_double",
        "blue_left_dashed_double",
        "yellow_single_solid",
        "yellow_double_solid",
    ],
    "vehicle_bike": [
        "white_single_solid",
        "white_double_solid",
        "blue_single_solid",
        "blue_double_solid",
        "blue_single_dashed",
        "blue_left_dashed_double",
        "blue_left_dashed_double",
        "yellow_single_solid",
        "yellow_double_solid",
    ],
    "vehicle_truck": [
        "white_single_solid",
        "white_double_solid",
        "blue_single_solid",
        "blue_double_solid",
        "blue_single_dashed",
        "blue_left_dashed_double",
        "blue_left_dashed_double",
        "yellow_single_solid",
        "yellow_double_solid",
    ],
    "vehicle_bus": [
        "white_single_solid",
        "white_double_solid",
        "yellow_single_solid",
        "yellow_double_solid",
    ],
    "danger": [
        "shoulder_single_solid",
        "shoulder_double_solid",
        "shoulder_left_dashed_double",
        "shoulder_left_dashed_double",
    ],
}

LANE_COLOR_MAP_MODEL = {
    "white_single_solid": (255, 0, 0),
    "white_double_solid": (255, 0, 0),
    "white_single_dashed": (255, 255, 255),
    "white_left_dashed_double": (255, 255, 255),
    "white_right_dashed_double": (255, 255, 255),
    "blue_single_solid": (255, 0, 0),
    "blue_double_solid": (255, 0, 0),
    "blue_single_dashed": (0, 0, 255),
    "blue_left_dashed_double": (0, 0, 255),
    "blue_left_dashed_double": (0, 0, 255),
    "yellow_single_solid": (255, 0, 0),
    "yellow_single_dashed": (255, 255, 255),
    "yellow_double_solid": (255, 0, 0),
    "yellow_left_dashed_double": (255, 255, 255),
    "yellow_left_dashed_double": (255, 255, 255),
    "shoulder_single_solid": (255, 253, 51),
    "shoulder_single_dashed": (255, 255, 255),
    "shoulder_double_solid": (255, 253, 51),
    "shoulder_left_dashed_double": (255, 253, 51),
    "shoulder_left_dashed_double": (255, 253, 51),
    "other": (255, 255, 255),
}
