from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('yolov8m.pt')
license_plate_detector = YOLO('models/11-08_best_weight_yolov8m_50_epoch.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

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
frame_number = -1
ret = True
while ret:
    frame_number += 1
    ret, frame = cap.read()
    if ret:
        results[frame_number] = {}
        # detect vehicles
        detections = coco_model.track(frame, persist=True)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection
            if int(class_id) in vehicles:
                vehicle_bounding_boxes = [[x1, y1, x2, y2, track_id, score]]

                for bbox in vehicle_bounding_boxes:
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]

                    # detect license plates
                    license_plates = license_plate_detector(roi)[0]

                    # process license plate
                    for license_plate in license_plates.boxes.data.tolist():
                        plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

                        # crop license plate
                        license_plate_crop = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]

                        # de-colorize
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                        # posterize
                        _, plate_threshold = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                        # OCR read license plate number
                        license_plate_text, license_plate_text_score = read_license_plate(plate_threshold)

                        # if plate could be read write results
                        if license_plate_text is not None:
                            results[frame_number][track_id] = {
                                'car': {
                                    'bbox': [x1, y1, x2, y2],
                                    'bbox_score': score
                                },
                                'license_plate': {
                                    'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                    'number': license_plate_text,
                                    'bbox_score': plate_score,
                                    'text_score': license_plate_text_score
                                }
                            }

# write results
write_csv(results, './results.csv')