# Men-in-Black

## 1. 개요
- 도로 교통 법규 위반 차량 감지
- 도로 위의 일상적인 교통 법규 위반, 특히 주요 도로에서의 끼어들기 같은 행위는 많은 운전자들에게 불편함과 안전 위험을 초래합니다. 하지만 위반 행위를 목격하여도, 주행 중 신고가 어려워 신고를 미루다 결국 하지 않게 되는 경우가 많습니다.
- 따라서 본 프로젝트에서 영상을 통해 교통 법규 위반을 자동으로 탐지하고 분류하는 모델을 개발하고자 했습니다.
- 이 모델을 다양한 법규 위반 상황을 식별하고 자동 신고 기능을 포함하여, 안전하고 공장한 도로 환경 조성에 기여하고자 합니다.

## 팀 구성 및 역할

### Line Violation Detection by [진한별](https://github.com/Moonbyeol)
- 프로젝트 계획 및 관리
- 데이터 수집 및 전처리
- 모델 설계 및 개발

### [Traffic Light Detection](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/main/Traffic%20Light) by [최우석](https://github.com/Wangws1004)
- 프로젝트 계획 및 관리
- 데이터 수집 및 전처리
- 모델 설계 및 개발

### [License Plate Recognition](https://github.com/SeSAC-Men-in-Black/Men-in-Black/tree/main/Automatic%20License%20Plate%20Recognition) by [신승엽](https://github.comsyshin0116)
- 프로젝트 계획 및 관리
- 데이터 수집 및 전처리
- 모델 설계 및 개발

## 2. 데이터셋 & 사용 툴

#### [COCO Dataset](https://cocodataset.org/#home)
- 330K images (>200K labeled)
- 1.5 million object instances
- 80 object categories
- Classes: Car, Motorcycle, Bus, Truck

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/99532836/98088b3c-1fed-4e4b-bde9-7617afaed7e7)


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
 
![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/99532836/af6e9b8e-045d-4c03-855b-bcdedfae3cdf)


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

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/99532836/8a34a5a8-51c2-4455-a723-1afdd4e986ca)

## 사용 툴
<img width="793" alt="image" src="https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/99532836/4508f00f-9b1b-4dca-941c-6966f17d5ec6">

## 프로젝트 일정 
![men_in_black_2023-11-16_01 43pm](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/99532836/f8f97160-0093-44aa-a3a4-2250958e438d)


