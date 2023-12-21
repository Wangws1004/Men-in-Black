![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/db7c842e-0a8a-42b4-90ad-88c1782c7658)# **VDE: Vehicle Distance Estimation from a Monocular Camera for Advanced Driver Assistance Systems**
![ezgif com-gif-maker](https://user-images.githubusercontent.com/98331298/171547569-da221132-a13e-4b5f-8437-59cad290d3b2.gif)  
- paper:https://www.mdpi.com/2073-8994/14/12/2657
- github: https://github.com/KyujinHan/Object-Depth-detection-based-hybrid-Distance-estimator


## 목표

- 단안 카메라를 사용하여 정확하고 효율적인 차량 거리 추정을 위한 프레임워크를 개발
- 거리 추정을 위한 프레임워크 제안 
- 객체 탐지 및 깊이 추정
- 거리 예측 모델 훈련


## 방법론

- 객체 탐지기 (DETR) : 단안 카메라로 촬영된 이미지에서 객체를 탐지하고, 각 객체에 대한 유형과 경계 상자의 좌표를 식별합니다.
- 깊이 추정기 (Global-Local Path Network) : 이미지에 대한 깊이 맵을 생성하여, 객체의 깊이 특징(예: 평균 깊이, 최소 깊이, 최대 깊이)을 추출합니다.
- 거리 예측기 (XGBoost, RF, LSTM) : 객체 유형, 경계 상자의 세부 정보 및 추출된 깊이 특징을 바탕으로 객체와 카메라 사이의 실제 거리를 예측합니다.
- 객체의 배경 픽셀을 제외한 20% 잘라낸 평균 깊이를 사용하여 거리를 더 정확하게 예측합니다.


## 모델 아키텍쳐

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/c5102726-34dd-4b41-90d7-8a6071c7fec4)

- Image input
- **DETR**과 **GLPDepth**에 각각 input
	- **DETR**에서는 class, object의 bounding box 좌표를 출력   
	- **GLPDepth**에서는 image의 depth을 생성
- Depth map 위에 bounding box를 overlapping
- Overlappning 된 bounding box를 기준으로 각 object의 depth 통계지표를 계산
- Bounding box, class information, depth의 통계 values를 LSTM에 넣어서 각 객체의 real distance를 계산

### 1. DETR

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/4cdda6c3-8324-4283-a32d-4a0f806416dc)
![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/75588172-3c01-44e7-a68c-c93db00b08d1)

- Image input
- CNN (Convolutional Neural Network) 사용하여 입력 이미지로부터 중요한 특징을 추출
- 추출된 특징 맵은 Transformer 모델로 전달
    - 인코더는 이미지 전체의 맥락을 분석
        
    - 디코더는 특정 객체 쿼리를 기반으로 각 객체의 클래스와 바운딩 박스(위치 및 크기)를 예측

### 2. GLPDepth structure

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/3c77aa23-5501-42e3-84a8-58de5850d120)

- Input image
- Global path
    - 이미지의 전체적인 맥락과 광범위한 특징을 파악하는 데 중점
    - Image를 Patch로 나눠서 1D로 flatten → 4번의 Transformer encoder block을 적용 → Block을 1/4, 1/8, 1/16, 1/32 사이즈로 scaling된 bottleneck feature 4개 output
- Local path  
    - 이미지 내의 세부적이고 구체적인 특징에 중점
    - embedding의 channel을 reduction한 후, upsampling을 진행 → 각 decoder stage마다 Global Path에서 나온 bottleneck feature를 SFF라는 network에 넣어서 embedding → Conv-ReLU-Conv와 sigmoid를 적용해서 최종적으로 depth-map


### 3. LSTM, RandomForest, XGBoost

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/23399896-7e2e-42dd-8903-ba92d113e5e7)
- LSTM과 뒤에 FFN을 붙여서 Real distance를 예측

## 데이터셋 전처리

- KITTI 데이터 사용
- KITTI 데이터셋 내의 각 객체의 경계 상자 좌표를 연구팀의 프레임워크에서 사용된 객체 탐지기로 식별된 좌표로 대체
- KITTI 데이터셋과 비교하여 두 경계 상자 간의 중첩 비율을 교집합 대 합집합(IoU) 함수를 사용하여 계산. 중첩 비율이 70% 이상이면 먼 객체의 경계 상자 제거, 70% 미만이면 중첩 영역 제외하고 깊이 특성 추
- 데이터셋을 시각적으로 검토하여 잘못된 객체 거리 정보를 가진 객체 제외


## 평가지표 및 결과

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/3aa7ccb7-3053-42c4-a443-76105017f6a0)
- 전체적인 성능은 LSTM이 뛰어났고, Car의 distance estimation 성능은 XGBoost가 우월했다.
![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/bfe4562f-10b3-4d8b-96c2-dde42ba0d7e8)

![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/4f25fe4e-00a4-494a-9bf8-9ca00ee13f2f)
![image](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140053617/822d02e0-ea4d-4751-9d81-e5c135d062f3)


## 실패 원인

- ValueError: X has 16 features, but StandardScaler is expecting 15 features as input.
- scaler 안의 특성 개수가 다른 것을 해결하지 못함



