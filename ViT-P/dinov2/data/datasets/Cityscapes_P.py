import torch
import os

# from torchvision.io import read_image
from PIL import Image 
import torchvision
import torchvision.transforms.v2 as transforms
# from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import pandas as pd
import cv2
from numpy import random
import glob
import numpy as np
import pickle
import json



class Cityscapes_P(Dataset):
    def __init__(self, split, n_points, image_size):
        if split=='train':
            self.augmentation = transforms.Compose([
                                    transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1), ratio=(0.75, 1.3333), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=True),
                                    transforms.RandomRotation((-60,60)),
                                    transforms.RandomHorizontalFlip(p=0.5)
                                ])
        else:
            self.augmentation = None


        if split=="train":
            with open('./datasets/cityscapes/train_cityscapes.pkl', 'rb') as file:
                self.root_dir = pickle.load(file)
            # self.root_dir = glob.glob('./datasets/cityscapes/leftImg8bit/' + split + '/*/*.png') +  glob.glob('./datasets/cityscapes/leftImg8bit/' + 'train_extra' + '/*/*.png')
        else:
            self.root_dir = glob.glob('./datasets/cityscapes/leftImg8bit/' + split + '/*/*.png')

            
        # self.root_dir = glob.glob('./cityscapes/leftImg8bit/' + split + '/*/*.png')
        print(len(self.root_dir))
        
        self.split = split
        self.n_points = n_points
        
        self.transform = torchvision.transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])


        self.trainId = {
          0 :      255 ,
          1 :      255 , 
          2 :      255 ,
          3 :      255 , 
          4 :      255 ,
          5 :      255 ,
          6 :      255 ,
          7 :        0 ,
          8 :        1 ,
          9 :      255 ,
          10 :      255 ,
          11 :        2 ,
          12 :        3 ,
          13 :        4 ,
          14 :      255 ,
          15 :      255 ,
          16 :      255,
          17 :        5 ,
          18 :      255 ,
          19 :        6 ,
          20 :        7 ,
          21 :        8 ,
          22 :        9 ,
          23 :       10 ,
          24 :       11 ,
          25 :       12 ,
          26 :       13 ,
          27 :       14 ,
          28 :       15 ,
          29 :      255 ,
          30 :      255 ,
          31 :       16 ,
          32 :       17 ,
          33 :       18 ,
          -1 :       -1 ,
        }
    

    
    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):

        img_pth = self.root_dir[idx]
        image = Image.open(img_pth).convert('RGB')

        if "train_extra" in img_pth:
            msk_pth =  img_pth.replace('leftImg8bit.png', 'gtCoarse_labelIds.png').replace('leftImg8bit', 'gtCoarse')
        else:
            msk_pth =  img_pth.replace('leftImg8bit.png', 'gtFine_labelIds.png').replace('leftImg8bit', 'gtFine')        

        
        # mask_np = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
        mask_np = Image.open(msk_pth)

        if self.augmentation:
            image0,mask_np0 = self.augmentation(image, mask_np)            
            mask_np0 = np.array(mask_np0)
            object_numbers = np.unique(mask_np0.reshape(-1), axis=0)
            object_numbers = [x for x in object_numbers if x not in [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]]
            
            if len(object_numbers) < 1:
                mask_np = np.array(mask_np)
                object_numbers = np.unique(mask_np.reshape(-1), axis=0)
                object_numbers = [x for x in object_numbers if x not in [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]]
            else:
                image = image0
                mask_np = mask_np0
        else:
            mask_np = np.array(mask_np)
            object_numbers = np.unique(mask_np.reshape(-1), axis=0)
            object_numbers = [x for x in object_numbers if x not in [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]]

        
        
        if self.transform:
            image = self.transform(image)
        

        # object_numbers = [x for x in object_numbers if x not in [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]]
        
        x = np.random.choice(object_numbers, size=self.n_points, replace=True)
        
        points = np.zeros((self.n_points,2))
        
        label = np.zeros((self.n_points))
        
        h,w =mask_np.shape
        j=0
        for i in x:
            ori = np.where( mask_np == i)
            rand = random.randint(ori[0].shape[0])
            points[j] = (ori[0][rand]/h, ori[1][rand]/w)
            label[j] = self.trainId[i]
            j+=1
        
        points = 2 * points - 1

        return {"image" : image,
                "points" : torch.from_numpy(points).to(dtype=torch.float32),
                "label" : torch.from_numpy(label).to(dtype=torch.float32),
               }

