import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import TLDataset
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
import datetime

class NIA_SEGNet_module(pl.LightningModule):
    # batch_size = 4096
    batch_size = 1

    def __init__(self):
        super().__init__()
        self.fcn = models.mobilenet_v2(pretrained=True)
        self.fcn.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fcn.last_channel, 5),
        )
        
    def forward(self, x):
        out = self.fcn(x)
        return out

    def get_loss(self, batch):
        x, y, _ = batch
        out = (self(x))
        loss = F.cross_entropy(out, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, img_path = batch
        out = (self(x))
        table = ["red", "yellow", "green", "greenleft","redleft"]
        imshow = True
#         imshow = False
        if imshow:
            # print(x.shape)
            for i in range(len(x)):
                img = x[i].permute(1,2,0)
                plt.imsave("output.png",img.cpu().numpy())
                print("label",table[y[i]],"pred",table[torch.argmax(out[i])])
                input()
        else:
            pred = torch.argmax(out,dim=1)
            for i in range(len(out)):
                print(img_path[i],"label : ",table[y[i]],"pred : ",table[pred[i]])
            confusion_mat = torch.zeros((5,5))
            for i in range(5):
                for j in range(5):
                    confusion_mat[i,j] = torch.sum((pred==i)*(y==j))
            return confusion_mat

    def test_epoch_end(self,outputs):
        confusion_mat_sum = 0
        for confusion_mat in outputs:
            confusion_mat_sum += confusion_mat
        aa,bb,cnt = 0,0,0
        for ii in range(5):
            if torch.sum(confusion_mat_sum[ii,:]) !=0 and torch.sum(confusion_mat_sum[:,ii]) != 0:
                aa += confusion_mat_sum[ii,ii]/torch.sum(confusion_mat_sum[ii,:]).float()
                bb += confusion_mat_sum[ii,ii]/torch.sum(confusion_mat_sum[:,ii]).float()
                cnt += 1
        aa /= cnt
        bb /= cnt
        f1 = (2*aa*bb/(aa+bb)).item()
        accuracy = torch.trace(confusion_mat_sum)/torch.sum(confusion_mat_sum).float()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log_dict({'val_loss': loss})
        return loss

    def validation_epoch_end(self, outputs):
        sum_loss = 0
        for loss in outputs:
            sum_loss += loss

    def val_dataloader(self):
        dataset = TLDataset(data_path="val")
        train_loader = DataLoader(dataset, batch_size = self.batch_size, num_workers=0)
        return train_loader

    def test_dataloader(self):
        dataset = TLDataset(data_path="sample")
        train_loader = DataLoader(dataset, batch_size = self.batch_size, num_workers=0)
        return train_loader

    def train_dataloader(self):
        dataset = TLDataset(data_path="train")
        train_loader = DataLoader(dataset, batch_size = self.batch_size, num_workers=0,shuffle=True)
        return train_loader
