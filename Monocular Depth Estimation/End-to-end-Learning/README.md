# End-to-end Learning for Inter-Vehicle Distance and Relative Velocity
# Estimation in ADAS with a Monocular Camera

- paper: [https://arxiv.org/pdf/2006.04082v2.pdf](https://arxiv.org/pdf/2006.04082v2.pdf)
- github: [https://github.com/ZhenboSong/mono_velocity](https://github.com/ZhenboSong/mono_velocity

## 요약
- 단안 카메라 (Monocular Camera)로 차량 간 거리 및 상대 속도를 추정
- U-net, DORN, M3D-RPN 으로 3D 객체 검출 방법을 통해 차량 간 거리 추정
- Flownet2, PWC-NET 으로 이미지의 흐름을 3D 모션 필드로 나타내 상대속도를 추정

  
	- U-net은 인코더(축소 경로)와 디코더(확장 경로)로 이루어진 U자형 구조를 가지고 있음. 인코더는 입력 이미지를 반복적으로 다운샘플링하여 고수준의 특징을 추출하고, 디코더는 이러한 특징을 업샘플링하여 원래 입력 이미지의 크기로 복원
  	- DORN(Depth Ordering Network)은 단일 이미지로부터 깊이를 예측하는 딥러닝 기반의 모델
  	- M3D-RPN(Monocular 3D Region Proposal Network)은 단일 이미지에서 객체의 3D bounding box를 예측하는 데 사용되는 딥러닝 기반의 모델
	- FlowNet 2.0은 컴퓨터 비전 분야에서 Flow를 예측하는 딥러닝 네트워크. 이 모델은 입력 이미지 간의 flow를 정확하게 추정하여 영상에서의 물체의 움직임이나 카메라의 움직임을 파악하는 데 사용
	- PWC-Net(Pyramidal Warping of Convolutional Features)은 영상 간 광학 흐름(Optical Flow)을 예측하기 위한 딥러닝 기반의 모델

## 목표와 도전 과제

- End-to-end 학습을 기반으로 1개의 단안 카메라 기반 차량 간 거리 및 상대 속도 추정
- 2개의 연속 프레임을 바탕으로 원근 왜곡의 영향을 완화하기 위한 차량 중심 샘플링 메커니즘 제안


## 방법론

- 도로 표면과 카메라 높이 사이의 기하학적 제약을 활용하여 거리 추정의 스케일 모호성 해결
- 2D 바운딩 박스가 3D 바운딩 박스 투영을 둘러싸고 있다고 가정하여 차량 거리 회귀 모델 제안
	- 감지된 2D 바운딩 박스의 균일한 폭과 높이를 입력값으로 사용
- 상대 속도는 시간 간격 동안 측정 카메라에 관측된 차량의 움직으로 추정
	- 속도가 빠르더라도 먼 거리에 있는 차량은 이미지의 차이가 작음
	- 속도가 느리더라도 근접 차량은 이미지의 차이가 큼

## 모델 아키텍처

- Monocular (이하 단안) depth와 3D 객체 검출 방법을 적용
- U-net 구조를 도입하여 지도 학습에 의해 깊이를 예측
- DORN 은 깊이를 순서형 회귀 문제로 간주
- M3D-RPN 은 3D 바운딩 박스가 2D 이미지 공간에서 생성된 Convolution 특징을 활용할 수 있는 3D 영역을 제안
- 상대속도는 이미지의 흐름을 3D 모션 필드로 나타냄
	- Flownet2, PWC-Net을 통해 여러 신경망을 stacking, warping
	- 이를 통해, 여러 신경망을 통합하여 가볍고 빠른 신경망 구현

![Screenshot 2023-12-20 at 2 57 16 PM](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/33dce05b-d1f6-4bf7-b442-88e025baf8d0)

<img width="1005" alt="Screenshot 2023-12-20 at 4 52 22 PM" src="https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/38f7bb53-64eb-43d3-842a-8553bc79dc22">

## 데이터셋과 사전 훈련

- 총 2가지 Dataset 활용 : Tusimple velocity(20 fps, 40 frame 비디오 시퀀스), KITTI
- FlyingChairs에 사전 훈련된 PWCNet을 기반으로 Feature map과 flow map을 통합하기 위해 7x7 ROI Align 사용
- Conv layer size : 3x3, 7x7
- Resize Image : 284 x 448
- 4가지 벡터를 병합
  (concatenating the geometric vector, deep feature vector and flow vector)
- 합쳐진 레이어들에 ReLU 활성화 함수를 사용
  (4 fully connected layers with ReLU activation function are employed to compute the distance and velocity)
- 전체 신경망을 PyTorch로 구현했으며, 손실함수는 ADAM을 사용
  (The whole network is implemented in PyTorch and trained end-to-end by optimizing the loss function with ADAM)


## Loss Function and Evaluation Metrics

![KakaoTalk_20231221_005632389](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/c87a5c50-7d63-4188-9576-ebdeb74bf43b)
- 이 표는 TuSimple 벤치마크에서의 속도 추정 결과
- "Rank1 [18]" 우승 알고리즘
- "ours org" 원본 이미지를 사용한 모델
- "ours full" End-to-end 차량 중심 모델
- 표에는 위치와 속도에 대한 평균 제곱 오차(MSE)로 비교
- "ours full" 모델은 가까운(near), 중간(medium), 먼(far) 거리에 대해 낮은 MSE 값을 보이며, 평균적으로 가장 낮은 오차(0.86)를 기록하여 다른 모델들에 비해 더 우수한 성능을 보여줌

![KakaoTalk_20231221_005632389_01](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/8582cd59-75f2-4cc5-8de1-1e857a772f64)
- 표 III는 TuSimple 벤치마크에서의 차량 거리 추정 결과를 나타냄
- "ours org"와 "ours full"은 각각 원본 이미지와 재샘플링된 이미지를 입력으로 사용한 모델
- 표 가장 아래 3개의 정확도 메트릭이 나타나 있으며, 가장 위에 오차 메트릭(첫 네 가지)은 낮을수록 좋고, 정확도 메트릭(마지막 세 가지)은 높을수록 성능이 좋음
- "ours full" 모델은 대체적으로 "ours org"보다 조금 더 높은 오차를 보이지만, 두 모델 모두 1.25, 1.25^2, 1.25^3의 정확도 임계값에서는 1.00의 완벽한 점수를 받음
- 이는 두 모델이 해당 임계값 이내에서 거리를 추정하는 데 매우 정확하다는 것을 의미
	- 오차 메트릭은 모델의 예측이 얼마나 정확한지를 평가하기 위해 사용되며, 이 값이 낮을수록 모델의 예측이 실제 값에 더 가깝다는 것을 의미
	- 일반적으로 사용되는 오차 메트릭으로는 절대 상대 오차(AbsRel), 제곱 상대 오차(SqRel), 루트 평균 제곱 오차(RMS), 로그 스케일에서의 RMS(RMSlog) 등이 있음
	- 정확도 메트릭은 모델 예측이 얼마나 종종 정확한 임계값 내에 있는지를 평가
	- 예측이 실제 값의 특정 배수 이내인 경우의 비율을 나타내는 임계값이 메트릭이 높을수록 더 많은 예측이 정확한 범위 내에 있음을 나타냄
  
![KakaoTalk_20231221_005632389_02](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/140369529/b831074b-b887-4137-8e8b-d4b597f03523)
- 표 IV와 V는 KITTI 데이터셋에서의 속도 추정과 거리 추정 결과를 보여줌
- 표 IV에서는 "ours full" 모델이 가까운 거리(MSE(near)), 중간 거리(MSE(medium)), 먼 거리(MSE(far)), 그리고 평균(MSE(average))에 대한 평균 제곱 오차를 나타냄
- 표 V에서는 "ours" 모델이 3DBox, DORN, Unsfm과 같은 다른 네트워크와 비교하여 거리 추정을 위한 다양한 메트릭(AbsRel, SqRel, RMS, RMSlog) 및 정확도 임계값을 기반으로 성능을 평가
- "ours" 모델은 거의 모든 메트릭에서 뛰어난 성능을 보이며, 특히 정확도 임계값에서는 가장 높은 점수를 기록
	- 3Dbox는 3차원 객체 탐지 네트워크
	- DORN (Deep Ordinal Regression Network)은 깊이 예측을 위한 네트워크로, 정확한 깊이 순서를 학습하는 것을 목표로 함
	- Unsfm은 또 다른 깊이 예측 네트워크로, 비정형적인 장면 구조에서도 성능을 발휘하도록 설계


## 결과
- 차량 추적이나 도로의 분할 없이 단안 렌즈의 두 이미지 시퀀스를 통해 공간적, 시간적 분석
- 해당 모델을 통해, 전방 충돌 경고 등의 분야에서 응용 가능

