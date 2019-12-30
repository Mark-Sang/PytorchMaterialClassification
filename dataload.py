import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms

#图片预处理
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((60,100)),
    transforms.ToTensor(),             #将图片转换为Tensor,归一化至[0,1]
    ])

#定义自己的正面图数据集合
class Front_viewPhotos(data.Dataset):
    def __init__(self,root):
        imgs=os.listdir(root)                   #所有图片的绝对路径
        imgs.sort(key=lambda x:int(x[:-4]))     #图片名从小到大排序
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transform=transform

    def __getitem__(self, index):
        img_path=self.imgs[index]
        pil_img=Image.open(img_path)
        pil_img = pil_img.crop((0,0,720,430))       #裁剪
        data=self.transform(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

#定义自己的侧面图数据集合
class Lateral_viewPhotos(data.Dataset):
    def __init__(self,root):
        imgs=os.listdir(root)                   #所有图片的绝对路径
        imgs.sort(key=lambda x:int(x[:-4]))     #图片名从小到大排序
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transform=transform

    def __getitem__(self, index):
        img_path=self.imgs[index]
        pil_img=Image.open(img_path) 
        pil_img = pil_img.crop((0,0,720,430))       #裁剪
        data=self.transform(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

