
# NIA 주행환경 정적객체 데이터셋을 위한 신호등 Classification

* <a href="https://arxiv.org/abs/1801.04381">Mobilenet V2</a> 기반 신호등 Classification의 학습, 예측을 위한 코드와 모델을 제공합니다.
* 주어진 이미지에서 빨간불/노란불/초록불/좌회전 신호가 있는지를 분류합니다.

## Demo
<!-- <img src="demo/input.png" alt="input" width="512" height=256/>
<img src="demo/output.png" alt="output" width="512" height=256/> -->

## Requirements

다음과 같은 python 라이브러리들이 필요합니다.

- numpy
- torch
- torchvision
- pytorch-lightning
- opencv-python
- matplotlib
- ujson


## Training
```
python train.py
```

## Inference
```
python test.py
```

## Pretrained Model
[Link]()


## Benchmark
| Model | Accuracy |
| --  | -- |
| Mobilenet V2  | 97.35% |

## Acknowledgements

- [Pytorch Mobilenet V2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
- [Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.](https://arxiv.org/abs/1801.04381)