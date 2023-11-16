# 모델 : Mask R-CNN / ResNet101
# 추가 적용 기법 : CARAFE / GRIOE / Seesawloss
# more details in https://mmdetection.readthedocs.io/en/latest/

# The new config inherits a base config to highlight the necessary modification
_base_ = "mask_rcnn_r101_fpn_2x_coco.py"


# ******************************************************** common config
gpu_ids = [0]
seed = 1111
work_dir = "./trained_models/lane_detection"
resume_from = "./pretrained/pretrained_lane_model.pth"
load_from = ""
# yapf:disable
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]
cudnn_benchmark = True
eval_interval = 500
evaluation = dict(interval=eval_interval, metric=["bbox", "segm"], save_best="segm_mAP")
dist_params = dict(backend="nccl")
log_level = "INFO"
workflow = [("train", 1)]
# runner = dict(type="EpochBasedRunner", max_epochs=10)
runner = dict(type="IterBasedRunner", max_iters=600000)
checkpoint_config = dict(interval=eval_interval, max_keep_ckpts=3)
# ******************************************************** common config

# ******************************************************** schedule config
# optimizer by mmdetection documents
# optimizer = dict(type="SGD", lr=1e-4, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type="Adam", lr=0.0003, weight_decay=0.0001)
optimizer_config = dict()

# learning policy by mmdetection documents
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[400000, 500000],
    min_lr=1e-7,
)

# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     min_lr_ratio=1e-5)
# ******************************************************** schedule config

# ******************************************************** dataset config
dataset_type = "CocoDataset"
data_source = "/data/"
ann_source = "/data/lane_cocostyle/"
classes = ("lane_blue", "lane_shoulder", "lane_white", "lane_yellow")

# default
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
# )
img_norm_cfg = dict(
    mean=[105.685, 99.015, 101.624], std=[65.58, 65.665, 67.324], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# train set
train_data_set = dict(
    type=dataset_type,
    img_prefix=data_source + "train/IMAGE",
    classes=classes,
    ann_file=ann_source + "train.json",
    pipeline=train_pipeline,
)
valid_data_set = dict(
    type=dataset_type,
    img_prefix=data_source + "valid/IMAGE",
    classes=classes,
    ann_file=ann_source + "valid.json",
    pipeline=test_pipeline,
)
test_data_set = dict(
    type=dataset_type,
    img_prefix=data_source + "test/IMAGE",
    classes=classes,
    ann_file=ann_source + "test.json",
    pipeline=test_pipeline,
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=train_data_set,
    val=valid_data_set,
    test=test_data_set,
)
# ******************************************************** dataset config

# ******************************************************** model config
"""
# 각 논문의 Ablation Study를 참고하여 AP가 높았던 parameter들을 적용해본다.

1. CARAFE (CARAFE: Content-Aware ReAssembly of FEatures)
    - compressed_channels (16, 32, 64, 128, 256) 중 32나 128이 AP50 기준 가장 높음
    - encoder kernel size k_encoder and reassembly kernel size k_up 값은 k_encoder = 5, k_up = 7일때 AP50 기준 가장 높다. 그런데 이게 cfg에서 어떤 변수인지...
    - 아마도 k_encoder = encoder_kernel이고, k_up = up_kernel인듯 한데 정확히 조사가 필요할듯?
    - 또한 논문의 저자는 k_encoder = k_up - 2로 잡는게 경험적으로 모든 case에 대해서 좋은 성능이 나왔다고 정리함. 이미 mmdetection에도 그렇게 되어있는듯
    - FPN뿐만 아니라 mask head에도 carafe를 적용하는것이 모든 AP 기준에서 가장 높음

2. GRIOE (A Novel Region of Interest Extraction Layer for Instance Segmentation)
    - GRIOE = Generic RoI Extractor LAYER
    - SELECTING FPN LAYERS (baseline, random, sum) 중 sum이 가장 AP 높음. 하지만 FPN은 이미 CARAFE를 적용 중
    - GRIOE는 4개 모듈(RoI pooler module, Pre-processing module, Aggregation module, Post-processing module)로 구성
    - Aggregation module(baseline, random, sum, sum+)에서 sum이 모든 AP 기준에서 가장 높음
    - Pre-processing module(baseline, conv1x1, conv3x3, conv5x5, Non-local, Attention)에서 conv5x5가 AP50 가장 높음
    - Post-processing module(baseline, conv1x1, conv3x3, conv5x5, Non-local, Attention)에서 attention이 AP50 가장 높음

3. SeesawLoss (Seesaw Loss for Long-Tailed Instance Segmentation)
    - Loss 구성 단계에서, Mask R-CNN 기준 Seesaw Loss + Norm Mask 방법이 가장 성능 높음
    - Norm Mask는 mask head에 추가하고 seesaw loss는 bbox head에서 cross entropy를 대체함
    - mitigation factor, compensation factor, and normalized linear activation 세 기법?을 모두
    - mitigation factor = p = 0.8이 가장 AP 높음
    - compensation factor = q = 2.0이 가장 AP 높음 / ImageNet-LT 데이터셋에서는 1.0이 AP 가장 높음
    - temperature term τ in Normalized Linear Activation. τ = 20 is the default setting이고 가장 AP 높음
    - Mask head 마지막 단에 Normalized Mask Predication 기법 추가. 1x1 conv이고 temperature term도 적용
"""
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    neck=dict(
        type="FPN_CARAFE",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=("conv", "norm", "act"),
        upsample_cfg=dict(
            type="carafe",
            up_kernel=7,
            up_group=1,
            encoder_kernel=5,
            encoder_dilation=1,
            compressed_channels=32,
        ),
    ),  # CARAFE: Content-Aware ReAssembly of FEatures
    roi_head=dict(
        bbox_roi_extractor=dict(
            type="GenericRoIExtractor",
            aggregation="sum",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            pre_cfg=dict(
                type="ConvModule",
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                padding=2,
                inplace=False,
            ),
            post_cfg=dict(
                type="GeneralizedAttention",
                in_channels=256,
                spatial_range=-1,
                num_heads=6,
                attention_type="0100",
                kv_stride=2,
            ),
        ),
        mask_roi_extractor=dict(
            type="GenericRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            pre_cfg=dict(
                type="ConvModule",
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                padding=2,
                inplace=False,
            ),
            post_cfg=dict(
                type="GeneralizedAttention",
                in_channels=256,
                spatial_range=-1,
                num_heads=6,
                attention_type="0100",
                kv_stride=2,
            ),
        ),  # GRIOE A novel Region of Interest Extraction Layer for Instance Segmentation
        bbox_head=dict(
            num_classes=4,
            loss_cls=dict(
                type="SeesawLoss", p=0.8, q=2.0, num_classes=4, loss_weight=1.0
            ),
        ),  # SeesawLoss Seesaw Loss for Long-Tailed Instance Segmentation
        mask_head=dict(
            num_classes=4,
            upsample_cfg=dict(  # CARAFE: Content-Aware ReAssembly of FEatures
                type="carafe",
                scale_factor=2,
                up_kernel=7,  # 5 -> 7로 고침
                up_group=1,
                encoder_kernel=5,  # 3 -> 5로 고침
                encoder_dilation=1,
                compressed_channels=32,
            ),  # 64 -> 32로 고침
            predictor_cfg=dict(type="NormedConv2d", tempearture=20),
        ),  # SeesawLoss의 Normalized Mask Predication
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        ),
    ),
)
# ******************************************************** model config
