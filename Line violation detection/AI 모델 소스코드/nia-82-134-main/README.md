# nia-82-134

차로 위반 영상 데이터를 이용한 학습 모델

## 환경 설정

### Using docker

1. Docker image build

    ```bash
    docker build -t nia-82-134 .
    ```

2. Run docker container

    ```bash
    docker run -it --gpus all \
    --name nia-test \
    -v /path/to/data:/data \
    nia-82-134
    ```

### Using conda and pip

1. Create conda virtual env

    ```bash
    conda create -n nia-82-134 python=3.8
    ```

2. Activate conda virtual env and Install cudatoolkit

    ```bash
    conda activate nia-82-134
    conda install -y cudatoolkit=11.0
    ```

3. Using pip to install packages

    ```bash
    pip install -r requirements.txt
    ```

## 모델별 학습 방법

### 1. 차량 인식 모델

#### 1.1 Config file

config file은 모델 구조, Datasets, Optimizer 등 학습 및 테스트에 필요한 파라미터를 구성

1.1.1 모델 구성
    mmdetection의 Mask R-CNN은 Backbone, Neck, RPN Head, ROI Head로 구성되어있고 각 단계마다 mmdetection 내부에서 구현되어있는 관련 기법을 적용했습니다.

* Detection Model : Mask R-CNN
* BackBone Network : ResNet101
* BackBone Pre-trained : torchvision://resnet101

1.1.2. 단계 별 적용 기법

* BackBone Network : ResNet101 (Pre-trained)
* Neck : 'FPN_CARAFE'
* RPN Head : default (기본 config)
* ROI Head : 'CARAFE', 'GROIE', 'SeesawLoss'

#### 1.2 학습 파라미터

학습 초기 파라미터들을 설정할 수 있습니다.

```python
# optimizer by mmdetection documents
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()

# learning policy by mmdetection documents
lr_config = dict(policy='step', min_lr=1e-7)

runner = dict(type='EpochBasedRunner', max_epochs=120)
resume_from = 'work_dirs/model_name/detection_model.pth'
```

* Optimizer
  * type : Gradient Descent 기법의 종류. ('SGD', 'Adam', 'Adamw')
        주의 : optimizer의 type을 'SGD'외 다른 기법으로 적용하면 학습이 불안정할 가능성이 있습니다.
        loss가 매우 큰 값으로 발산한다거나, NaN값으로 계산되는 등 여러 사례가 있습니다.
  * lr : learning rate
  * momentum : Gradient Descent 기법의 파라미터
  * weight_decay : Gradient Descent 기법의 파라미터

* lr_config
  * policy : learning rate schedule의 종류 ('Step', 'CosineAnnealing', 'poly' 등등...)
  * min_lr : learning rate을 감소시킬 수 있는 최소한.

* runner
  * type : 'EpochBasedRunner'로 고정
  * max_epoch : 학습을 진행할 epoch 수

* resume_from : 이전에 학습했던 모델을 이어서 학습할때, 학습을 진행했던 모델 파일의 경로
* load_from : mmdetection에서 제공하는 Pre-trained 모델 파일의 경로. <https://mmdetection.readthedocs.io/en/v2.19.0/model_zoo.html> 에서 다운로드 받을 수 있습니다.

#### 1.3 학습 시작

```bash
python train_vehicle_detection.py \
--config ./configs/vehicle_detection_config.py
```

#### 1.4 성능 측정

```bash
python eval_vehicle_detection.py \
--config ./configs/vehicle_detection_config.py \
--ckpt ./best_models/vehicle_detection_model.pth
```

#### 1.5 시각화

```bash
python viz_vehicle_detection.py \
--config ./configs/vehicle_detection_config.py \
--ckpt ./best_models/vehicle_detection_model.pth \
--input_image ./path/to/image.jpg \
--save_dir ./viz_output/vehicle_detection
```

---

### 2. 차선 인식 모델

#### 2.1 Config file

차량 인식 모델과 동일

#### 2.2 학습 파라미터

학습 초기 파라미터들을 설정할 수 있습니다. 차량 인식 모델과 동일

#### 2.3 학습 시작

```bash
python train_lane_detection.py \
--config ./configs/lane_detection_config.py
```

#### 2.4 성능 측정

```bash
python eval_lane_detection.py \
--config ./configs/lane_detection_config.py \
--ckpt ./best_models/lane_detection_model.pth
```

#### 2.5 시각화

```bash
python viz_lane_detection.py \
--config ./configs/lane_detection_config.py \
--ckpt ./best_models/lane_detection_model.pth \
--input_image ./path/to/image.jpg \
--save_dir ./viz_output/lane_detection
```

---

### 3. 위반 탐지 모델

#### 3.1 마스킹 이미지 생성

```bash
python build_masking_images.py \
--img_save_dir /data/vlt_cls_data \ # 마스킹 이미지 저장 경로
--annot_root /data # 원본 JSON과 이미지가 있는 경로
```

#### 3.2 모델 학습 설정 파일

```yaml
data_dir: "/data/vlt_cls_data" # 마스킹 이미지의 최상위 디렉토리 경로
model_save_dir: "./trained_models/vlt_cls_model" # 학습된 모델을 저장할 디렉토리 경로
load_model: # fine-tuning할 경우 불러올 모델의 경로
batch_size: 1024
lr: 0.001
lr_step_size: 5 # step learning scheduler를 사용할 때 지정할 step size
lr_gamma: 0.5 # learning rate scheduler의 lr 감소 비율
num_epochs: 20 # 학습할 epoch 수
```

#### 3.3 학습 시작

```bash
python train_vlt_cls_model.py \
--config_file ./configs/vlt_cls_model_config.yaml
```

#### 3.4 위반 탐지 모델 성능 측정

```bash
python eval_vlt_cls_model.py \
--config_file ./configs/vlt_cls_model_config.yaml
```

#### 3.5 위반 탐지 모델 시각화

```bash
python viz_vlt_cls_inference.py \
--config_file ./configs/vlt_cls_model_config.yaml
```
