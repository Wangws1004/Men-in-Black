from module import NIA_SEGNet_module
import pytorch_lightning as pl

model = NIA_SEGNet_module.load_from_checkpoint(checkpoint_path="D:/신호등도로표지판 인지 영상(수도권)/신호등-도로표지판 인지 영상(수도권)/Men-in-Black/신호등 도로표지판 인지 영상(수도권)/신호 상태 분류 프로토타입 모델/trafficlight/epoch=5-step=113.ckpt")
trainer = pl.Trainer(devices=1, accelerator="gpu") 
# trainer = pl.Trainer(gpus=8, distributed_backend="ddp")
# trainer = pl.Trainer(gpus=2, distributed_backend="ddp")
print(trainer.test(model))
