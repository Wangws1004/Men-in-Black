from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv, interpolate_bounding_boxes, process_and_visualize_license_plate, visualize
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import ast

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8m.pt')
license_plate_detector = YOLO('models/11-10_best_weight_yolov8m_140_epochs.pt')

# load video
video_path = './sample/road_sample_license_plate.mp4'
video_name = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

person = [0]
vehicles = [1, 2, 3, 5, 7]
traffic_signs = [9, 11]

class_id_dict = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorbike',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign'
}

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        # print('detected license plate:', license_plates)
        for license_plate in license_plates.boxes.data.tolist():
            print('detected license plate boxes:', license_plate)
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                processed_license_plate = process_and_visualize_license_plate(license_plate_crop)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(processed_license_plate)
                print('license_plate_text:', license_plate_text)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}


print("results: \n", results)
# write results
csv_file_name = f'./sample/{video_name}.csv'
write_csv(results, csv_file_name)

# Add missing data

# Load the CSV file
with open(csv_file_name, 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open(f'./sample/{video_name}_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)


# Visualize
visualize(video_path)
