import os
import argparse
import torch
import torchvision
import numpy as np
from torch.utils import data
import copy
import argparse       
import cv2          
import torch.nn as nn            
from torch.nn.functional import normalize       
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
import math     
import torchvision.datasets as datasets

from scipy.special import logsumexp
import time


class cifar_10_data(datasets.ImageFolder):
    def __init__(self, data_path ='', size=224, s=1.0, mean=None, std=None, blur=False):
        super(cifar_10_data, self).__init__(data_path)
        self.train_transform = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=23))
        self.train_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor(),
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)
   
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        return self.train_transform(image), self.train_transform(image), index, label

class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample
