_base_ = [
    "base/mask_rcnn_r50_fpn.py",
    "base/coco_instance.py",
    "base/schedule_2x.py",
    "base/default_runtime.py",
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)
