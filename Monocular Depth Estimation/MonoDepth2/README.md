# Monodepth2, Digging Into Self-Supervised Monocular Depth Estimation
- paper : [https://arxiv.org/pdf/1806.01260.pdf](https://arxiv.org/pdf/1806.01260.pdf)
- github : [https://github.com/nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2)
## 목표와 도전 과제
__[목표]__
- 단안 카메라 이미지만을 사용하여 깊이를 추정(다른 센서와 같은 추가적인 하드웨어 없이 3차원 환경을 이해할 수 있도록 연구)
- 자율 주행 자동차, 로봇공학, 증강 현실 등 분야에서 응용을 목표로 함  
  
__[도전과제]__
1. 스케일 불변성 (Scale Ambiguity) : 단안(mono) 이미지에서 물체의 실제 크기를 알기 어려움. 스테레오(stereo) 시스템과 달리 단안(mono) 시스템은 물체까지 절대적인 거리를 직접적으로 측정할 수 없음. 이로 인해 추정된 깊이의 스케일이 실제 환경과 다를 수 있으며, 이를 해결하기 위한 방법이 필요함
2. 동적 객체 (Dynamic Objects) : 
