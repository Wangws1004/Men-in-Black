# Men-in-Black

## 1. 개요
- 도로 교통 법규 위반 차량 감지
- 도로 위의 일상적인 교통 법규 위반, 특히 주요 도로에서의 끼어들기 같은 행위는 많은 운전자들에게 불편함과 안전 위험을 초래합니다. 하지만 위반 행위를 목격하여도, 주행 중 신고가 어려워 신고를 미루다 결국 하지 않게 되는 경우가 많습니다.
- 따라서 본 프로젝트에서 영상을 통해 교통 법규 위반을 자동으로 탐지하고 분류하는 모델을 개발하고자 했습니다.
- 이 모델을 다양한 법규 위반 상황을 식별하고 자동 신고 기능을 포함하여, 안전하고 공장한 도로 환경 조성에 기여하고자 합니다.

## 2. 프로젝트 구성 및 담당자

### [Line Violation Detection](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/074ad63391bab45290966de5b0f9d747f9a252ae/Line%20violation%20detection) by [진한별](https://github.com/Moonbyeol)

### [Traffic Light Recognition](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/main/Traffic-Light-Recognition) by [최우석](https://github.com/Wangws1004)

### [License Plate Recognition](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/main/Automatic%20License%20Plate%20Recognition) by [신승엽](https://github.comsyshin0116)


# Traffic Light Recognition

## 진행 과정:

- Model: Yolov8m epochs = 200, batch = 32, lrf = 0.001
    
- Dataset: [Roboflow - Traffic light Computer Vision Project](https://universe.roboflow.com/trafficlightdetect/traffic-light-ke5b5)
    
    - 1,000 images (Train / Valid / Test = 701 / 199 / 100)           
        - Augmentations (Train Image 4206 장으로 데이터 증강)
            - RandomResizedCrop(640)  # 이미지 크기를 무작위로 자르고 640x640 크기로 조절
            - RandomHorizontalFlip(),  # 50% 확률로 좌우 반전
            - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상, 대비, 채도 및 색조를 무작위로 조절
            - RandomRotation(10),  # 최대 10도까지 무작위로 회전
            - RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 무작위로 이동, 크기 조절, 기울임 변환
            - GaussianBlur(kernel_size=3),  # 가우시안 블러 적용


<img width="857" alt="Screenshot 2023-12-21 at 1 54 37 PM" src="https://github.com/Wangws1004/WS_Project/assets/140369529/e5afd348-70b5-484a-9195-56a986e5f8b0">



- Training:
    
hyper parameters:

`Ultralytics YOLOv8.0.73 🚀 Python-3.10.9 torch-2.0.1 CUDA:0 (Tesla T4, 14972MiB)
yolo/engine/trainer: task=detect, mode=train, model=yolov8m.pt, data=/content/Traffic_light/data.yaml, epochs=200, patience=50, batch=32, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=None, exist_ok=False, pretrained=False, optimizer=SGD, verbose=True, seed=0, deterministic=True, single_cls=False, image_weights=False, rect=False, cos_lr=False, close_mosaic=0, resume=False, amp=True, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, vid_stride=1, line_thickness=3, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.001, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, v5loader=False, tracker=botsort.yaml, save_dir=runs/detect/train21
Overriding model.yaml nc=80 with nc=4`



model summary:


<img width="765" alt="Screenshot 2023-12-21 at 1 46 12 PM" src="https://github.com/Wangws1004/WS_Project/assets/140369529/e423bcc2-ccdc-4beb-b1c9-b8c7f79fc46c">



## WandB
- Confusion matrix
![Traffic_light_confusion_matrix](https://github.com/Wangws1004/WS_Project/assets/140369529/2ff90079-120c-4010-8142-2a06ed8c00cb)

- F1 curve
![Traffic_light_F1_curve](https://github.com/Wangws1004/WS_Project/assets/140369529/87105929-1d6b-4c23-8b39-99047e1e7b64)

- P curve
![Traffic_light_P_curve](https://github.com/Wangws1004/WS_Project/assets/140369529/516b5558-3293-439e-b141-739a23e90820)

- R curve
![Traffic_light_R_curve](https://github.com/Wangws1004/WS_Project/assets/140369529/6a8277b4-7657-4aeb-9a80-6b0043a995e1)

- PR curve
![Traffic_light_PR_curve](https://github.com/Wangws1004/WS_Project/assets/140369529/a34f456a-1c17-409b-8ddc-8981ff7d323f)

- Result
![Traffic_light_result](https://github.com/Wangws1004/WS_Project/assets/140369529/65ed54e7-51c8-4372-9292-aa03c3bd5b29)



## 결과
- Train image

![Traffic_light_train_image_sample](https://github.com/Wangws1004/WS_Project/assets/140369529/eceb3a9a-fb2e-41cc-9200-043aa545d870)

- Valid Image

![Traffic_light_valid_image_sample](https://github.com/Wangws1004/WS_Project/assets/140369529/fe6e84e2-43a1-488f-a316-50b9554923f5)
 


