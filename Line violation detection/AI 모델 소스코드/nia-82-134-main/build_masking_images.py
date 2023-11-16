# coding: utf-8
import argparse
import json
import os
from glob import glob

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from constants import (
    DANGER_COLOR,
    LANE_COLOR_MAP,
    NEW_SIZE,
    NORMAL_COLOR,
    VEHICLE_COLOR_MAP,
    VEHICLE_LIST,
    VLT_COLOR,
)

parser = argparse.ArgumentParser()
parser.add_argument("--img_save_dir", default="/data/vlt_cls_data")
parser.add_argument("--annot_root", default="/data")
args = parser.parse_args()


def draw_and_save(
    save_dir, info_list, lane_img, org_img, lane_img_arr, basename, vlt_type, vlt_color
):
    for i, car in enumerate(info_list):
        lane_img = lane_img.copy()
        blank_img = Image.new("RGB", org_img.size)
        draw = ImageDraw.Draw(blank_img)
        color = VEHICLE_COLOR_MAP[car[0]]
        draw.polygon(car[2], fill=color)

        car_img_arr = np.array(blank_img.resize(NEW_SIZE, Image.NEAREST))
        car_img_arr = np.mean(car_img_arr, axis=2)
        car_img_arr = np.where(car_img_arr > 0, True, False)

        draw = ImageDraw.Draw(lane_img)
        draw.polygon(car[2], fill=color)
        lane_img = np.array(lane_img.resize(NEW_SIZE, Image.NEAREST))
        lane_img[lane_img_arr & car_img_arr] = vlt_color
        lane_img = Image.fromarray(lane_img)
        lane_img.save(os.path.join(save_dir, f"{basename}_{vlt_type}_{i}.jpg"))


def main():
    for split in ["test", "valid", "train"]:
        annotation_list = glob(
            os.path.join(args.annot_root, split, "ANNOTATION/*.json")
        )
        curr_save_dir = os.path.join(args.img_save_dir, split)
        print(f"Curr dir {split} has {len(annotation_list)} annotations.")
        for annot in tqdm(annotation_list):
            try:
                with open(annot, "rb") as f:
                    annot_data = json.load(f)
                basename = os.path.splitext(os.path.basename(annot))[0]

                # get lane and vehicle info
                lane_list = []
                vlt_list = []
                danger_list = []
                normal_list = []
                for d in annot_data["data_set_info"]["data"]:
                    d = d["value"]
                    label = d["object_Label"]
                    if "type" in label.keys():
                        ctg = d["object_Label"]["type"]
                        attr = d["object_Label"]["attribute"]
                    elif "vehicle_type" in label.keys():
                        ctg = d["object_Label"]["vehicle_type"]
                        attr = d["object_Label"]["vehicle_attribute"]
                    else:
                        ctg = d["object_Label"]["lane_type"]
                        attr = d["object_Label"]["lane_attribute"]
                    points = d["points"]
                    polys = []
                    for p in points:
                        polys.append((p["x"], p["y"]))
                    if ctg not in VEHICLE_LIST:
                        lane_list.append((ctg, attr, polys))
                    else:
                        if attr == "violation":
                            vlt_list.append((ctg, attr, polys))
                        elif attr == "danger":
                            danger_list.append((ctg, attr, polys))
                        else:
                            normal_list.append((ctg, attr, polys))

                # draw lane
                image_path = annot.replace("ANNOTATION", "IMAGE").replace("json", "jpg")
                input_img = Image.open(image_path)
                img = Image.new("RGB", input_img.size)
                draw = ImageDraw.Draw(img)

                for lane in lane_list:
                    color = LANE_COLOR_MAP[lane[0]][lane[1]]
                    draw.polygon(lane[2], fill=color)
                lane_img_arr = np.array(img.resize(NEW_SIZE, Image.NEAREST))
                lane_img_arr = np.mean(lane_img_arr, axis=2)
                lane_img_arr = np.where(lane_img_arr > 0, True, False)

                # Save normal images
                save_dir = os.path.join(curr_save_dir, "normal")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                draw_and_save(
                    save_dir,
                    normal_list,
                    img,
                    input_img,
                    lane_img_arr,
                    basename,
                    "normal",
                    NORMAL_COLOR,
                )

                # Save danger images
                save_dir = os.path.join(curr_save_dir, "danger")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                draw_and_save(
                    save_dir,
                    danger_list,
                    img,
                    input_img,
                    lane_img_arr,
                    basename,
                    "danger",
                    DANGER_COLOR,
                )

                # Save vlt images
                save_dir = os.path.join(curr_save_dir, "violation")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                draw_and_save(
                    save_dir,
                    vlt_list,
                    img,
                    input_img,
                    lane_img_arr,
                    basename,
                    "violation",
                    VLT_COLOR,
                )

            except ValueError as e:
                print(f"Error in {annot} {e}")
            except KeyError as e:
                print(f"Error in {annot} {e}")
            except:
                print(f"ETC Error in {annot}")


if __name__ == "__main__":
    main()
