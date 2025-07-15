# coding: utf-8
'''
Code is referred from https://github.com/klintan/pytorch-lanenet
delete the one-hot representation for instance output

This code defines a custom PyTorch dataset class called RailDataset designed for railway track segmentation tasks. 
It loads images and their corresponding labels (both binary and instance segmentation masks) for training a neural network.
RailDataset class:
- Loads images and corresponding binary + instance masks.
- Converts color images properly.
- Processes masks to binary (rail vs background) and instance labels.
- Masks out the top half of the images (ROI).
- Supports applying transformations to both image and masks.
- Provides shuffled dataset samples for training.
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from dataloader.locoEYE_transformers import ToTensor
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

class RailDataset(Dataset):
    def __init__(self, dataset, n_labels=3, transform=None, target_transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.n_labels = n_labels

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip().split(',')
                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        '''
        loads the image and its two types of labels for a single sample, processes and cleans the masks, 
        optionally transforms them, and returns them together.
        '''
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) == len(self._gt_img_list)
        img = cv2.cvtColor(cv2.imread(self._gt_img_list[idx]), cv2.COLOR_BGR2RGB)  # load image as np array RGB
        label_instance_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED)
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        # Only lower half ROI
        height = label_binary.shape[0]
        label_binary[:height // 2, :] = 0
        label_instance_img[:height // 2, :] = 0

        sample = (img, label_binary, label_instance_img)

        if self.transform:
            # print("Transform class:", type(self.transform))
            # print("Transform list (if Compose):", getattr(self.transform, 'transforms', 'Not a Compose object'))
            sample = self.transform(sample)

        # sample now is tuple of tensors (image, mask, instance_mask)
        return sample
