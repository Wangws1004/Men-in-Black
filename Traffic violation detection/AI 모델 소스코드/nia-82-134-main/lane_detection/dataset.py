import json
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LaneDataset(Dataset):
    def __init__(self, root, split="train"):
        if split == "train":
            self.I_H = 800
            self.I_W = 1333
        else:
            self.I_H = 800
            self.I_W = 1333

        self.img_list = glob(os.path.join(root, split, "IMAGE/*.jpg"))
        self.label_path = [
            i.replace("IMAGE", "ANNOTATION").replace("jpg", "json")
            for i in self.img_list
        ]

        self.len = len(self.img_list)

    def __getitem__(self, index):
        while True:
            img_path = self.img_list[index]
            try:
                img = Image.open(img_path)
                break
            except:
                with open("error_files.txt", "a") as errlog:
                    errlog.write(str(index) + ": " + img_path + "\n")
                    index = index + 1

        w, h = img.size
        label_path = self.label_path[index]
        with open(label_path, "r") as f:
            json_data = json.load(f)
        img_tensor = transforms.functional.to_tensor(
            transforms.functional.resized_crop(
                img, h - w // 2, 0, w // 2, w, (self.I_H, self.I_W)
            )
        )
        target_map = self.make_gt_map(json_data, w, h)

        return img_tensor, torch.LongTensor(target_map), img_path

    def __len__(self):
        return self.len

    def make_gt_map(self, json_data, original_w, original_h):

        target_map = np.zeros((self.I_H, self.I_W), dtype=np.int32)
        annotation = json_data["data_set_info"]["data"]
        y_offset = original_h - original_w // 2

        for item in annotation:
            label = item["value"]["object_Label"]
            if "lane_type" in label.keys():
                obj_class = label["lane_type"]
                obj_lab_att = label["lane_attribute"]
            else:
                continue
            if obj_class[5:] == "white":
                pos = item["value"]["points"]
                poly_points = np.array(
                    [
                        (
                            [
                                pt["x"] * self.I_W / original_w,
                                (pt["y"] - y_offset)
                                * self.I_H
                                / (original_h - y_offset),
                            ]
                        )
                        for pt in pos
                    ]
                ).astype(np.int32)
                if obj_lab_att == "single_solid":
                    cv2.fillPoly(target_map, [poly_points], 1)
                elif obj_lab_att == "single_dashed":
                    cv2.fillPoly(target_map, [poly_points], 1)
                elif obj_lab_att == "double_solid":
                    cv2.fillPoly(target_map, [poly_points], 1)
                elif obj_lab_att == "left_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 1)
                elif obj_lab_att == "right_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 1)

            elif obj_class[5:] == "blue":
                pos = item["value"]["points"]
                poly_points = np.array(
                    [
                        (
                            [
                                pt["x"] * self.I_W / original_w,
                                (pt["y"] - y_offset)
                                * self.I_H
                                / (original_h - y_offset),
                            ]
                        )
                        for pt in pos
                    ]
                ).astype(np.int32)
                if obj_lab_att == "single_solid":
                    cv2.fillPoly(target_map, [poly_points], 2)
                elif obj_lab_att == "single_dashed":
                    cv2.fillPoly(target_map, [poly_points], 2)
                elif obj_lab_att == "double_solid":
                    cv2.fillPoly(target_map, [poly_points], 2)
                elif obj_lab_att == "left_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 2)
                elif obj_lab_att == "right_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 2)

            elif obj_class[5:] == "yellow":
                pos = item["value"]["points"]
                poly_points = np.array(
                    [
                        (
                            [
                                pt["x"] * self.I_W / original_w,
                                (pt["y"] - y_offset)
                                * self.I_H
                                / (original_h - y_offset),
                            ]
                        )
                        for pt in pos
                    ]
                ).astype(np.int32)
                if obj_lab_att == "single_solid":
                    cv2.fillPoly(target_map, [poly_points], 3)
                elif obj_lab_att == "single_dashed":
                    cv2.fillPoly(target_map, [poly_points], 3)
                elif obj_lab_att == "double_solid":
                    cv2.fillPoly(target_map, [poly_points], 3)
                elif obj_lab_att == "left_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 3)
                elif obj_lab_att == "right_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 3)

            elif obj_class[5:] == "shoulder":
                pos = item["value"]["points"]
                poly_points = np.array(
                    [
                        (
                            [
                                pt["x"] * self.I_W / original_w,
                                (pt["y"] - y_offset)
                                * self.I_H
                                / (original_h - y_offset),
                            ]
                        )
                        for pt in pos
                    ]
                ).astype(np.int32)
                if obj_lab_att == "single_solid":
                    cv2.fillPoly(target_map, [poly_points], 4)
                elif obj_lab_att == "single_dashed":
                    cv2.fillPoly(target_map, [poly_points], 4)
                elif obj_lab_att == "double_solid":
                    cv2.fillPoly(target_map, [poly_points], 4)
                elif obj_lab_att == "left_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 4)
                elif obj_lab_att == "right_dashed_double":
                    cv2.fillPoly(target_map, [poly_points], 4)

        return target_map
