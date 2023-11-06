from module import NIA_SEGNet_module
import pytorch_lightning as pl

import torch
print(torch.cuda.is_available())

model = NIA_SEGNet_module.load_from_checkpoint(checkpoint_path="D:/Men-in-Black/Traffic Light/Traffic Light Status/Traffic Light/epoch=5-step=113.ckpt")
trainer = pl.Trainer(devices=1, accelerator="gpu") 
# trainer = pl.Trainer(gpus=8, distributed_backend="ddp")
# trainer = pl.Trainer(gpus=2, distributed_backend="ddp")
print(trainer.test(model))
print("Success")