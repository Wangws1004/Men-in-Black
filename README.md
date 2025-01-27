# Men-in-Black

## 1. 개요
- 도로 교통 법규 위반 차량 감지
- 도로 위의 일상적인 교통 법규 위반, 특히 주요 도로에서의 끼어들기 같은 행위는 많은 운전자들에게 불편함과 안전 위험을 초래합니다. 하지만 위반 행위를 목격하여도, 주행 중 신고가 어려워 신고를 미루다 결국 하지 않게 되는 경우가 많습니다.
- 따라서 본 프로젝트에서 영상을 통해 교통 법규 위반을 자동으로 탐지하고 분류하는 모델을 개발하고자 했습니다.
- 이 모델을 다양한 법규 위반 상황을 식별하고 자동 신고 기능을 포함하여, 안전하고 공장한 도로 환경 조성에 기여하고자 합니다.

## 2. 프로젝트 구성 및 담당자

### [Line Violation Detection](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/074ad63391bab45290966de5b0f9d747f9a252ae/Line%20violation%20detection) by [진한별](https://github.com/Moonbyeol)
<details>
<summary>Details</summary>
<br>
</details>

### [Traffic Light Recognition](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/main/Traffic-Light-Recognition) by [최우석](https://github.com/Wangws1004)
<details>
<summary>Details</summary>
<br>
</details>

### [License Plate Recognition](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/main/Automatic%20License%20Plate%20Recognition) by [신승엽](https://github.com/syshin0116)
<details>
<summary>Details</summary>

## 진행 과정:

1. 차량 감지(Vehicle Detection)
    
2. 번호판 감지(License Plate Detection)
    
3. OCR(Optical Character Recognition)
    

## 1. 차량 감지(Vehicle Detection)

- Model: Yolov8n, Yolov8m
    
- Dataset: COCO Dataset
    
    - 330K images (>200K labeled)
        
    - 1.5 million object instances
        
    - 80 object categories
        
- Classes: Car, Motorcycle, Bus, Truck
    

YOLO model structure

![](https://i.imgur.com/eFgToyo.png)


### 차량 트래킹(Object Tracking)

- model: Sort
    
    - A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences
        
- [GitHub - abewley/sort: Simple, online, and realtime tracking of multiple objects in a video sequence.](https://github.com/abewley/sort)
    

###   2. 번호판 감지(License Plate Detection)

- Model: Yolov8m 50 epoch, 120epoch
    
- Dataset: \[Roboflow][License Plate Recognition Object Detection Dataset (v4, resized640_aug3x-ACCURATE) by Roboflow Universe Projects](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4 "https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4")
    
    - 24242 images
        
        - Augmentations
            
            - Flip: Horizontal
                
            - Crop: 0% Minimum Zoom, 15% Maximum Zoom
                
            - Rotation: Between -10° and +10°
                
            - Shear: ±2° Horizontal, ±2° Vertical
                
            - Grayscale: Apply to 10% of images
                
            - Hue: Between -15° and +15°
                
            - Saturation: Between -15% and +15%
                
            - Brightness: Between -15% and +15%
                
            - Exposure: Between -15% and +15%
                
            - Blur: Up to 0.5px
                
            - Cutout: 5 boxes with 2% size each
                
- Training:
    

hyper parameters:

`task=detect, mode=train, model=yolov8m.pt, data=/content/License_plate_recognition/dataset/License-Plate-Recognition-4/data.yaml, epochs=500, patience=50, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=license_plate_detection_yolov8m, name=None, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, vid_stride=1, stream_buffer=False, line_width=None, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=license_plate_detection_yolov8m/train`

model summary:

`from n params module arguments 0 -1 1 1392 ultralytics.nn.modules.conv.Conv [3, 48, 3, 2] 1 -1 1 41664 ultralytics.nn.modules.conv.Conv [48, 96, 3, 2] 2 -1 2 111360 ultralytics.nn.modules.block.C2f [96, 96, 2, True] 3 -1 1 166272 ultralytics.nn.modules.conv.Conv [96, 192, 3, 2] 4 -1 4 813312 ultralytics.nn.modules.block.C2f [192, 192, 4, True] 5 -1 1 664320 ultralytics.nn.modules.conv.Conv [192, 384, 3, 2] 6 -1 4 3248640 ultralytics.nn.modules.block.C2f [384, 384, 4, True] 7 -1 1 1991808 ultralytics.nn.modules.conv.Conv [384, 576, 3, 2] 8 -1 2 3985920 ultralytics.nn.modules.block.C2f [576, 576, 2, True] 9 -1 1 831168 ultralytics.nn.modules.block.SPPF [576, 576, 5] 10 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest'] 11 [-1, 6] 1 0 ultralytics.nn.modules.conv.Concat [1] 12 -1 2 1993728 ultralytics.nn.modules.block.C2f [960, 384, 2] 13 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest'] 14 [-1, 4] 1 0 ultralytics.nn.modules.conv.Concat [1] 15 -1 2 517632 ultralytics.nn.modules.block.C2f [576, 192, 2] 16 -1 1 332160 ultralytics.nn.modules.conv.Conv [192, 192, 3, 2] 17 [-1, 12] 1 0 ultralytics.nn.modules.conv.Concat [1] 18 -1 2 1846272 ultralytics.nn.modules.block.C2f [576, 384, 2] 19 -1 1 1327872 ultralytics.nn.modules.conv.Conv [384, 384, 3, 2] 20 [-1, 9] 1 0 ultralytics.nn.modules.conv.Concat [1] 21 -1 2 4207104 ultralytics.nn.modules.block.C2f [960, 576, 2] 22 [15, 18, 21] 1 3776275 ultralytics.nn.modules.head.Detect [1, [192, 384, 576]] Model summary: 295 layers, 25856899 parameters, 25856883 gradients`

optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 77 weight(decay=0.0), 84 weight(decay=0.0005), 83 bias(decay=0.0)

Image sizes: 640 train, 640 val

#### WandB

![](https://i.imgur.com/wKFGARx.png)

![](https://i.imgur.com/ZwzZCZh.png)

![](https://i.imgur.com/iGsTw9O.png)

![](https://i.imgur.com/AKzo4Tz.png)

![](https://i.imgur.com/UKG85j0.png)

![](https://i.imgur.com/1B8dgOW.png)

## 3. OCR(Optical Character Recognition)

Model: EasyOCR

Preprocessing steps:

1. **Grayscale Conversion**: This simplifies the image by removing color information, making further processing faster and focusing on intensity.
    
2. **Contrast Enhancement with CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Improves the contrast of the image, making details more distinct, especially useful in varying lighting conditions.
    
3. **Gaussian Blur**: Reduces noise and smoothes the image, which can help in reducing false edges detected in the subsequent edge detection step.
    
4. **Canny Edge Detection**: Identifies edges in the image. This is useful for finding the boundaries of objects, in this case, the license plate.
    
5. **Finding Contours and Perspective Transformation**: Identifies contours in the image and, if a rectangular contour (assumed to be the license plate) is found, applies a perspective transformation to get a front-facing view of the license plate.
    

Original Image:

![](https://i.imgur.com/63v2mMO.png)


Detected Car:

![](https://i.imgur.com/50zAgWN.png)


Grayscale:

![](https://i.imgur.com/3h2XYY4.png)

CLAHE:

![](https://i.imgur.com/Nt70a3p.png)

Gaussian Blur:

![](https://i.imgur.com/I0Cg8wH.png)


Canny Edge Detection:

![](https://i.imgur.com/vEbTsXy.png)

## Attempts and Failure:

- Tracking cars with yolov8:
    
    - worse outputs compared to Sort, took longer time→ attempted at early stages, improved output expected
        
- Clips from Dashboard cam:
    
    - Car and License Plates were well detected, but video quality too low for OCR
        
    - phenomenon occurred more frequently when relative speed of vehicle was faster
        

## Room for Improvements:

- Try variety of Object Detection models for comparison
    
- Try variety of OCR models for comparison(TesseractOCR, PaddleOCR)
    
- Enhance Video Quality for better detection and recognition
    
- Try Segmentation
</details>

## 3. 데이터셋 & 사용 툴

#### [COCO Dataset](https://cocodataset.org/#home)
- 330K images (>200K labeled)
- 1.5 million object instances
- 80 object categories
- Classes: Car, Motorcycle, Bus, Truck

![](https://i.imgur.com/X6FioAe.png)




#### [\[Roboflow\]License Plate Recognition Object Detection Dataset (v4, resized640_aug3x-ACCURATE)](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
- 24242 images
- 데이터 증강(Augmentation)
  - Flip: Horizontal 
  - Crop: 0% Minimum Zoom, 15% Maximum Zoom 
  - Rotation: Between -10° and +10° 
  - Shear: ±2° Horizontal, ±2° Vertical 
  - Grayscale: Apply to 10% of images 
  - Hue: Between -15° and +15° 
  - Saturation: Between -15% and +15% 
  - Brightness: Between -15% and +15% 
  - Exposure: Between -15% and +15% 
  - Blur: Up to 0.5px 
  - Cutout: 5 boxes with 2% size each
 
![](https://i.imgur.com/CqP6mNG.png)


 #### [\[AIHUB\]차로 위반 영상 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=628)
 - 80,000장 이미지
 - 원시 데이터 포맥 예시(동영상)
   - MP4 포맷의 동영상 클립
   - FHD 해상도
   - 초당 5 프레임
 - 원천데이터 포맷 예시(이미지 추출 및 비식별화 이후)
   - JPG 포맥 이미지 실 예시
   - FHD 해상도
   - 비식별화 처리(사람얼굴, 자동차 번호판, 개인 전화번호 등)

![](https://i.imgur.com/8PRmusV.png)

## 4. 사용 툴

![](https://i.imgur.com/I0kSdG1.png)
## 5. 프로젝트 일정 

![](https://i.imgur.com/35Cr1cR.png)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/0c31325c-acec-4954-a768-e079714aa469)



