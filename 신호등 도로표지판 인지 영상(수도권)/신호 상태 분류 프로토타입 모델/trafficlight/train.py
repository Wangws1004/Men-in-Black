from module import NIA_SEGNet_module
import pytorch_lightning as pl
import torch
import random
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

torch.manual_seed(777)
random.seed(777)

model = NIA_SEGNet_module()
# trainer = pl.Trainer(gpus=1)

trainer = pl.Trainer(gpus=[0], distributed_backend="ddp", callbacks=[EarlyStopping(monitor='val_loss',patience=10)])
trainer.fit(model)
