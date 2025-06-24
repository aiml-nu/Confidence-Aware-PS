#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data_utils
import scipy.io as sio
from PIL import Image
from torchvision.transforms import ToTensor
import os
import cv2
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def default_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:,:,0]
    return img

class Blobby_data(data_utils.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, loader=default_loader):
 
        self.imgs = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        root = '/home/crw9838/PS/data_gen/dataset/'
        label_x_, label_d_, label_n_ = self.imgs[index]
        label_x, label_d, label_n = root + label_x_, root + label_d_, root + label_n_
        gt_depth = cv2.imread(label_d,  cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)[:,:,0]
        gt_normal = cv2.imread(label_n,  cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

        gt_depth[gt_depth==65504] = 0
        gt_depth = gt_depth/374.0 
        gt_mask = gt_depth.copy()
        gt_mask = np.where((gt_mask > 0) & (gt_mask < 500), 1, 0)
        imgs = np.zeros((96, 512, 512), dtype=np.float32)
        
        for i in range(96):
            img_temp = self.loader(label_x+str(i).zfill(2)+'.png') * gt_mask
            imgs[i, :, :] = np.array(img_temp, dtype=np.float32)
        imgs = imgs/np.max(imgs)
        imgs = torch.from_numpy(imgs)

        return imgs,gt_depth.astype(float),gt_normal.astype(float).transpose(2,0,1), gt_mask.astype(float)

    def __len__(self):
        return len(self.imgs)