import os
import random
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image
lights = ["red","yellow","green","greenleft","redleft"]
labels = {"red":0, "yellow":1, "green":2, "greenleft":3, "redleft":4}
class TLDataset(Dataset):
    def __init__(self,data_path):
        trans = transforms.Compose([transforms.Resize((20,70)),transforms.ToTensor()])
        # if True:
        # if not os.path.exists(data_path.split("/")[-1]+".pt"):
        self.inputs = []
        self.labels = []
        self.img_list = []
        for light in lights:
            lightpath = os.path.join(data_path, light)
            if not os.path.isdir(lightpath):
                os.makedirs(lightpath)
            imglist = os.listdir(lightpath)
            for imgname in imglist:
                self.img_list.append(os.path.join(lightpath,imgname))
                img = Image.open(os.path.join(lightpath,imgname))
                newimg = img.resize((70,20))
                self.inputs.append(trans(newimg))
                self.labels.append(labels[light])
            # torch.save((self.inputs, self.labels, self.img_list),data_path.split("/")[-1]+".pt")
        # else:
        #     self.inputs, self.labels, self.img_list = torch.load(data_path.split("/")[-1]+".pt")
        self.data_path = data_path
        self.len = len(self.inputs)
        print("dataset size : ", self.len)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index], self.img_list[index]

    def __len__(self):
        return self.len
        


if __name__ == "__main__":
    ld = TLDataset(data_path="/media/NIA_jeju12_2TB/NIA_trafficlight_cropped/val")
    for img, target in ld:
        plt.imshow(img.permute((1,2,0)))
        plt.show()
        print(img.shape, target)
