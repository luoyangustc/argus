#!/usr/bin/python
#-*- encoding:utf-8 -*-
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

import os
import cv2
import numpy as np
import utils.util as util
import utils.keys as keys


class imageDataset(Dataset):
    def __init__(self, mapping, transform=None, target_transform=None, Test=False):
        """Initialization for image Dataset.
        args
        root (string): directory of images
        mapping (string): file of mapping filename and its labels

        """
        self.transform = transform
        self.target_transform = target_transform
        self.images = list()
        self.labels = list()
        self.Test = Test
        
        with open(mapping) as f:
            pair_list = f.readlines()
            self.nSample = len(pair_list)

        for pair in pair_list:
            items = pair.strip().split()
            img = items[0]
            # label = items[1] # No blank in the middle of the label string.
            label = ' '.join(items[1:])
            self.images.append(img)
            self.labels.append(keyFilte(label, keys.alphabet))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        if self.Test:
            return (img, self.labels[index], self.images[index])
        return (img, self.labels[index])


class resizeNormalize(object):

    def __init__(self, maxW, interpolation=Image.BILINEAR):
        self.maxW = maxW
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img,imgW):
        img = Image.fromarray(img).convert('L')
        img = Image.fromarray(255 - np.array(img))
        padding = (0, 0, self.maxW - imgW, 0)
        img = ImageOps.expand(img, border=padding, fill='black')
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class graybackNormalize(object):

    def __init__(self):
        return

    def __call__(self, img):
        img = Image.fromarray(255 - np.array(img))
        return img


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=True, min_ratio=1, Test=False):
        """
        args:
                imgH: can be divided by 32
                maxW: the maximum width of the collection
                keep_ratio: 
                min_ratio:
        """
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.Test = Test

    def __call__(self, batch):
        if self.Test:
            images, labels, srcs = zip(*batch)
        else:
            images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(self.min_ratio * imgH, imgW)  # assure imgW >= imgH

        transform = resizeNormalize(imgW, imgH)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        if self.Test:
            return images, labels, srcs
        return images, labels


def keyFilte(text, alphabet):
    valid_char = []
    for char in text:
        if alphabet.find(char) != -1:
            valid_char.append(char)
    if len(valid_char) == 0:
        for i in range(5):
            valid_char.append(random.choice(alphabet))

    return ''.join(valid_char)
