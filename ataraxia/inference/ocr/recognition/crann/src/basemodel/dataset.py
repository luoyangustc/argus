#!/usr/bin/python
#-*- encoding:utf-8 -*-
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
# import lmdb
import six
import sys
import os
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
from . import py_util
import logging

class resizeNormalize(object):

    def __init__(self, maxW, imgH, interpolation=Image.BILINEAR):
        self.imgH = imgH
        self.maxW = maxW
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        ratio = 1.0 * img.size[0] / img.size[1]
        imgW = int(self.imgH * ratio)
        img = img.resize((imgW, self.imgH), self.interpolation)
        padding = (0, 0, self.maxW - imgW, 0)
        img = ImageOps.expand(img, border=padding, fill='black')
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class resizeNormalizePadding(object):
    
    def __init__(self, imgH, interpolation=Image.BILINEAR):
        self.imgH = imgH
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        w, h = img.size
        ratio = 1.0 * w / h 
        imgW = int((self.imgH - 2) * ratio)
        # logging.critical(self.imgH)
        # logging.critical(imgW)
        rzimg = img.resize((max(imgW,1), max(self.imgH - 2,1)), self.interpolation)
        padding = (3, 1, 3, 1) 
        rzimg = ImageOps.expand(rzimg, border=padding, fill='black')
        img = self.toTensor(rzimg)
        img.sub_(0.5).div_(0.5)
        return img
