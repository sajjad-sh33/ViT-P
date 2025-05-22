# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

import albumentations as A
import cv2
import random
import torch

logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        num_points = 4
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.num_points = num_points

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.geometric_augmentation_global1 = A.Compose(
            [
                A.RandomResizedCrop(
                    (self.global_crops_size,self.global_crops_size), scale=self.global_crops_scale, interpolation=cv2.INTER_CUBIC
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,p=0.8),
                A.ToGray(p=0.2),    
                A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0),
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['key_numbers'], remove_invisible=True))
        
        
        self.geometric_augmentation_global2 = A.Compose(
            [
                A.RandomResizedCrop(
                    (self.global_crops_size,self.global_crops_size), scale=self.global_crops_scale, interpolation=cv2.INTER_CUBIC
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,p=0.8),
                A.ToGray(p=0.2),    
                A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.2),
                A.Rotate(limit=(-90, 90), p=0.9),
                A.Solarize(threshold_range=(0.5, 0.5), p=0.2),
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['key_numbers'], remove_invisible=True))
        
        # self.geometric_augmentation_local  = A.Compose(
        #     [
        #         A.RandomResizedCrop(
        #             (self.local_crops_size,self.local_crops_size), scale=self.local_crops_scale, interpolation=cv2.INTER_CUBIC
        #         ),
        #         A.HorizontalFlip(p=0.5),
        #         A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,p=0.8),
        #         A.ToGray(p=0.2),    
        #         A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5)
        #     ], keypoint_params=A.KeypointParams(format='xy', label_fields=['key_numbers'], remove_invisible=True))
        
        
        
        
        
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )


    def __call__(self, image):
        output = {}



        length, width,_= image.shape 
        distance = random.randint(100, 200)
        keypoints = generate_points(width, length, distance)
        key_numbers  = list(range(len(keypoints)))
        # global crops:

        while True:
            global_crop_1 = self.geometric_augmentation_global1(image=image, keypoints=keypoints, key_numbers=key_numbers)
            transformed_image1 = self.normalize(global_crop_1['image'])
            transformed_keypoints1 = global_crop_1['keypoints']
            transformed_key_numbers1 = global_crop_1['key_numbers']
            
            global_crop_2 = self.geometric_augmentation_global2(image=image, keypoints=keypoints, key_numbers=key_numbers)
            transformed_image2 = self.normalize(global_crop_2['image'])
            transformed_keypoints2= global_crop_2['keypoints']
            transformed_key_numbers2 = global_crop_2['key_numbers']
    
    
            common_numbers = set(transformed_key_numbers1).intersection(transformed_key_numbers2)
            if len(sorted(common_numbers)) >= self.num_points:
                break
        
        selected_points = random.sample(sorted(common_numbers), self.num_points)
        

        
        # Find indices in both lists
        indices1 = find_indices(transformed_key_numbers1, selected_points)
        indices2 = find_indices(transformed_key_numbers2, selected_points)
        tensor_keypoints1 = torch.tensor([transformed_keypoints1[i] for i in indices1])
        normalized_tensor_keypoints1 = 2*(tensor_keypoints1/self.global_crops_size)-1
        
        tensor_keypoints2 = torch.tensor([transformed_keypoints2[i] for i in indices2])
        normalized_tensor_keypoints2 = 2*(tensor_keypoints2/self.global_crops_size)-1


        output["global_crops"] = [transformed_image1]

        # global crops for teacher:
        output["global_crops_teacher"] = [transformed_image1]
        output["global_points"] = [normalized_tensor_keypoints1]
        
        
        output["local_crops"] = [transformed_image2]
        output["local_points"] = [normalized_tensor_keypoints2]


        output["offsets"] = ()

        return output




def generate_points(width, length, d):
    points = []
    for x in range(d, width, d):
        for y in range(d, length, d):
            points.append((x, y))
    return points

def find_indices(lst, points):
    return [lst.index(point) for point in points if point in lst]