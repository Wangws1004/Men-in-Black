# Monodepth2, Digging Into Self-Supervised Monocular Depth Estimation
- paper : [https://arxiv.org/pdf/1806.01260.pdf](https://arxiv.org/pdf/1806.01260.pdf)
- github : [https://github.com/nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2)
## 목표와 도전 과제
__[목표]__
- 단안 카메라 이미지만을 사용하여 깊이를 추정(다른 센서와 같은 추가적인 하드웨어 없이 3차원 환경을 이해할 수 있도록 연구)
- 자율 주행 자동차, 로봇공학, 증강 현실 등 분야에서 응용을 목표로 함  
  
__[도전과제]__
1. 스케일 불변성 (Scale Ambiguity) : 단안(mono) 이미지에서 물체의 실제 크기를 알기 어려움. 스테레오(stereo) 시스템과 달리 단안(mono) 시스템은 물체까지 절대적인 거리를 직접적으로 측정할 수 없음. 이로 인해 추정된 깊이의 스케일이 실제 환경과 다를 수 있으며, 이를 해결하기 위한 방법이 필요함
2. 동적 객체 (Dynamic Objects) : 움직이는 물체는 주변 환경에 비해 상대적으로 다른 움직임 패턴을 보여줌. 이는 단안(mono) 깊이 추정 모델에 혼선을 줄 수 있으며, 동적 객체를 처리하는게 큰 도전임.
3. 텍스처가 없는 영역 (Textureless Regions) : 매끄럽고 텍스처가 없는 영역은 깊이 정보를 추출하기 어렵게 만듬. 이러한 영역은 깊이 추정 시, 노이즈를 발생시킬 수 있음.
4. 재투영 불일치(Reprojection Inconsistencies) : 단안 시스템에서는 다른 시점의 이미지로부터 깊이를 추론할 때, 재투영(reprojection) 과정에서 불일치가 발생할 수 있음. 이는 가려진 영역(occlusions) 또는 반사(reflections)와 같은 복잡한 장면 특성 때문임.
5. 일관성 유지(Consistency Maintenance) : 시간에 따라 캡처된 연속적인 이미지들 사이에서 깊이의 일관성을 유지하는 것은 어려운 작업임.
## 방법론
Monodepth2는 단안 깊이 추정(Monocular depth estimation)을 위한 자기 지도 학습(self-supervision)방법을 발전 시킨 연구로써 몇 가지 핵심 기술을 도입하여 단안 이미지로부터 깊이 정보를 추정하는 성능을 향상 시킴  
1. 재투영 손실(Reprojection loss)  
여러 시점에서 캡쳐된 이미지들을 이용하여 한 이미지에서 다른 이미지로 픽셀을 재투영할 때 일관성을 측정함.  
재투영 손실은 모델이 다른 뷰에서 보이는 같은 장면의 픽셀을 정확하게 매핑할 수 있도록 도와줌
2. 자동 마스킹(Auto-Masking)  
카메라 또는 물체 움직임이 없는 경우, 깊이 추정이 어려워짐.  
그래서 자동 마스킹 기능이 이러한 정지된 프레임을 자동으로 감지하여 깊이 학습에서 제외 시킴.
3. 다중 스케일 매칭(Multi-Scale Matching)  
깊이 추정은 다양한 스케일에서 일관성을 유지해야함. 그러므로 다중 스케일에서 외관 일치를 측정하여 깊이 추정 정확도를 높임.
4. 최소 재투영 손실(Minimum Reprojection loss)  
재투영 후보 중에서 가장 낮은 손실을 갖는 것을 선택하여 가려짐이나 동적 객체로 인한 잠재적인 오류를 최소화 시킴.
5. 깊이와 포즈 네트워크(Depth and Pose Networks)  
깊이 추정을 위한 네트워크(Depth Network)와, 이미지 시퀀스 간의 카메라 움직임(Pose Network)을 추정하기 위한 두 네트워크를 사용함.
6. 엣지 인식 손실(Edge-Aware Smoothness loss)  
깊이 맵 연속성을 유지하면서도 엣지 부분에서는 급격한 깊이 변화를 허용하는 손실 함수를 적용하여, 엣지를 유지하면서 전체적인 매끄러움을 개선함

## 모델 아키텍처
효율적인 단안 깊이 추정(Monocular depth estimation)을 위해 깊이 네트워크(Depth Networ)와 포즈 네트워크(Pose Network)를 사용함.  
두 네트워크는 각각 이미지로부터 깊이 맵을 예측하고, 연속된 이미지 프레임 사이 카메라 또는 물체 움직임(포즈)을 추정하는 역할을 함.  

__1. 깊이 네트워크(Depth Network)__  
- U-Net 아키텍처로 인코더-디코더 구조를 가지고 있음
- 인코더
    - 이미지로부터 고차원 특징을 추출하는 역할을 하며, pretrain된 ResNet을 사용함.
- 디코더
    - 인코더에서 추출된 특징을 바탕으로 픽셀 단위의 깊이 맵을 생성함. 이 과정에서 skip connections(스킵 연결) 사용하여 인코더의 저차원 특징과 디코더의 고차원 특징을 결합함으로써, 세부적인 텍스처와 엣지 정보를 보존함.
    - 디코더 출력은 다중 스케일의 깊이 맵이며, 깊이 추정을 더욱 정밀하게 말들기 위해 사용됨.
![Untitled (8)](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/ace7a429-97a6-4b3b-b96c-a3c9994abd2a)
  
__2. 포즈 네트워크(Pose Network)__  
- 연속된 이미지 프레임 사이의 카메라 또는 물체 움직임을 추정함.
- ResNet18로 구성되어있고, 여러 연속 프레임을 입력받아 카메라 이동(포즈)을 6DoF(Degree of Freedom, 3축의 회전과 3축의 이동)로 예측함.
- 포즈 네트워크의 출력은 깊이 네트워크와 함께 사용되어 재투영 손실(Reprojection loss)을 계산하는데 필요한 정보를 제공함.
![Untitled (9)](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/a6d7038f-9eab-49f8-b56d-fff9b3887def)

두 네트워크는 자기 지도 학습(self-supervision)을 통해 동시에 훈련되며, 이 과정에서 재투영 손실(Reprojection loss), 자동 마스킹(Auto-Masking), 최소 재투영 손실(Minimum Reprojection loss), 엣지 인식 손실(Edge-Aware Smoothness loss) 등 다양한 손실 함수를 적용하여 깊이를 잘 추정하고자 함  

__3. 학습과정__  
![Untitled (10)](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/f2c48922-89c1-4c32-96fe-6e30100f610b)
(a)인 Depth 네트워크에서 Depth 추정하고 (b)인 Pose 네트워크에서 6-DoF 상대 위치를 추정함. 최종적으로 (c)인 appearance loss 에 두 추정 값이 사용됨

## 데이터셋
- KITTI Eigen Split  
  - KITTI 데이터셋에서 특정한 방식으로 훈련, 검증, 테스트 세트를 분할한 데이터로 Eigen et al.이 제안한 분할 방식을 따름.
  - 이전 연구에서도 사용된 표준적인 방식임.
- Pre-processing  
  - Zhou et al.의 방법을 따라 정지 프레임을 제거.
  - 정지 프레임은 주변 환경 변화없이 카메라만 움직인 경우를 의미하며, 이러한 프레임들은 깊이 학습에 부정적인 영향을 미칠 수 있음 그러므로 제거!!
- Monocular and Stereo Data
  - Monodepth2는 단안(monocular)데이터와 스테레오(stereo)데이터를 모두 사용할 수 있음.
  - 단안 데이터만을 사용하는 경우와 스테레오 데이터를 함께 사용하는 경우(mono+stereo)를 구분하여 실험함.

## 평가지표 및 결과
모델 성능 평가를 위해 다음과 같은 표준적인 깊이 추정 평가 지표가 사용됨.
1. 절대 상대 오차(Absolute Relative Error, Abs Rel)  
  예측된 깊이와 실제 깊이 사이의 상대적 차이를 평균하여 측정함.
2. 제곱 상대 오차(Squared Relative Error, Sq Rel)  
  예측된 깊이와 실제 깊이의 차이를 제곱한 값을 사용하여 오차를 계산함.
3. 루트 평균제곱 오차(Root Mean Squared Error, RMSE)  
  예측된 깊이와 실제 깊이 사이의 오차를 제곱하여 평균한 후 제곱근을 취함.
4. RMSE 로그(RMSE log)  
  깊이 값의 로그를 취한 후 RMSE를 계산함.
5. 정확도 지표(Threshold Accuracy)  
  예측된 깊이 값이 실제 값의 특정 배율 내에 있을 비율을 측정하고, 일반적인 임계값은 1.25, 1.25^2, 1.25^3 등으로 설정됨.
  
![Quantitative results](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/7c1c4f7e-9168-45d3-8866-2717325b2071)
본 방법론은 monocular 방식으로 설계가 되었지만, Stereo 성능에서도 높은 결과를 달성한 것이 흥미로움.  
위의 결과는 post-processing 을 진행하지 않았을때의 결과임.  
본 결과에 post-processing까지 적용한다면 아래 표처럼 성능이 조금 상승하게 됨.  
![Effect of post-processing result](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/7a14f48b-3265-42e8-89c2-59682764e563)  

그리고 본 논문에서 Ablation study 결과는 아래와 같음.  
![Ablation study](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/1681fea1-20eb-4aec-8657-6d64ba502c21)  
제안한 방법론을 하나씩 적용하고 적용하지 않았을 때 엄청난 성능차이가 있는거 같아 보이진 않음  

다른 모델들과 비교했을때 Monodepth2가 Ground Truth와 가장 depth값이 유사한 것으로 보임
![monodepth_Result](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/28aafc37-0733-4b4d-8837-beaa5ae9b7ab)

## 결론
1. Monodepth2는 기존의 self-supervision 기반의 단안 깊이 추정(monocular depth estimation) 방법보다 더 좋은 성능을 보임
2. 그러나 복잡한 장면이나 빛의 반사가 심한 영역이나, 비램버트 표면(Lambertian surface, 모든 방향에서 보아도 똑같은 밝기로 보이는 표면)등에 대한 깊이 추정의 정확도가 낮을 수 있음
  ![화면 캡처 2023-12-21 143809](https://github.com/SeSAC-Men-in-Black/Men-in-Black/assets/83795101/13b2e53b-a5b9-408a-acee-603c5c5bd5fa)
