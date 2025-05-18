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



class COCO_P(Dataset):

    def __init__(self, split, n_points, image_size):
        if split=='train':
            self.augmentation = transforms.Compose([
                                    # transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1), ratio=(0.75, 1.3333), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=True),
                                    transforms.RandomRotation((-60,60)),
                                    transforms.RandomHorizontalFlip(p=0.5)
                                ])
        else:
            self.augmentation = None
            
        self.split = split
        self.root_dir =  glob.glob('./coco/' + split +'2017'+ '/*.jpg')             
        self.n_points = n_points
        
        self.transform = torchvision.transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])


    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):

        img_pth = self.root_dir[idx]
        image = Image.open(img_pth)
        splited = img_pth.split('/')
        splited = splited[-1].split('.')
        msk_pth = './coco/panoptic_semseg_' + self.split +'2017/'+ splited[0] +'.png'
        mask_np = Image.open(msk_pth)

        if self.augmentation:
            image0,mask_np0 = self.augmentation(image, mask_np)            
            mask_np0 = np.array(mask_np0)
            object_numbers = np.unique(mask_np0.reshape(-1), axis=0)
    
            if len(object_numbers)==1 and (255 in object_numbers):
                mask_np = np.array(mask_np)
                object_numbers = np.unique(mask_np.reshape(-1), axis=0)
            else:
                image = image0
                mask_np = mask_np0
        else:
            mask_np = np.array(mask_np)
            object_numbers = np.unique(mask_np.reshape(-1), axis=0)

        
        if self.transform:
            image = self.transform(image)


        # object_numbers = np.unique(mask_np.reshape(-1), axis=0)


        # if len(object_numbers) > 1 :
        #     if 255 in object_numbers:
        object_numbers = np.delete(object_numbers,np.where(object_numbers==255))    
        
        x = np.random.choice(object_numbers, size=self.n_points, replace=True)
        
        points = np.zeros((self.n_points,2))
        
        label = np.zeros((self.n_points))
        
        h,w =mask_np.shape
        j=0
        for i in x:
            ori = np.where( mask_np == i)
            rand = random.randint(ori[0].shape[0])
            points[j] = (ori[0][rand]/h, ori[1][rand]/w)

            label[j] = i

            j+=1
        
        points = 2 * points -1

        return {"image" : image,
                "points" : torch.from_numpy(points).to(dtype=torch.float32),
                "label" : torch.from_numpy(label).to(dtype=torch.float32),
               }
