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



class Objects365_P(Dataset):
    def __init__(self, split, n_points, image_size ):
        # if split=='train':
        #     self.augmentation = transforms.Compose([
        #                             #transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1), ratio=(0.75, 1.3333), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=True),
        #                             transforms.RandomRotation((-60,60)),
        #                             transforms.RandomHorizontalFlip(p=0.5)
        #                         ])
        # else:
        #     self.augmentation = None
        self.split = split
        
        if split=='train':
            file_path = './Objects365/dsdl_Det_full/set-train/train_samples.json'  
        else:
            file_path = './Objects365/dsdl_Det_full/set-val/val_samples.json'
        
        with open(file_path, 'r') as file:
            root_dir = json.load(file)
            
        root_dir = root_dir['samples'] 

        cured_list = []
        for i in range(len(root_dir)):
            sample = root_dir[i]
            if len(sample['annotations']) > 0 :
                cured_list.append(sample)

        self.root_dir = cured_list
        print(len(cured_list))
        self.n_points = n_points
        
        self.transform = torchvision.transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        sample = self.root_dir[idx]
        img_pth = "./Objects365/" + sample['media']['media_path']
        image = Image.open(img_pth).convert('RGB')

        # if self.augmentation:
        #     image0,mask_np0 = self.augmentation(image, mask_np)            
        #     mask_np0 = np.array(mask_np0)
        #     object_numbers0 = np.unique(mask_np0.reshape(-1), axis=0)
    
        #     if len(object_numbers0)==1 and (0 in object_numbers0):
        #         mask_np = np.array(mask_np)
        #     else:
        #         image = image0
        #         mask_np = mask_np0

        # else:
        #     mask_np = np.array(mask_np)


        
        object_numbers = len(sample['annotations'])
        if self.transform:
            image = self.transform(image)
        
        
        x = np.random.choice(range(object_numbers), size=self.n_points, replace=True)
        
        points = np.zeros((self.n_points,2))
        
        label = np.zeros((self.n_points))
        
        _, h,w = image.shape
        j=0
        for i in x:
            y_min, x_min, L1, L2 = sample['annotations'][i]['bbox']
            L2 = max(L2, 1)
            L1 = max(L1, 1)
            x_min = min(x_min,0)
            y_min = min(y_min,0)
            px = random.randint(x_min,x_min+L2+1)
            py = random.randint(y_min,y_min+L1+1)
            points[j] = (px/h, py/w)
            label[j] = sample['annotations'][i]['category_id'] - 1
            j+=1
        
        points = 2 * points - 1

        return {"image" : image,
                "points" : torch.from_numpy(points).to(dtype=torch.float32),
                "label" : torch.from_numpy(label).to(dtype=torch.float32),
               }
