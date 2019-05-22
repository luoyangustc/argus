#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os
from .Machersolution import Matcher
import numpy as np 
import json
import base64
import requests
import sys

Debug = False

class IDCARDCLASS(object):
    def __init__(self,templatefront,templateback):
        templatefrontimg = cv2.imread(templatefront)
        templatebackimg = cv2.imread(templateback)
        self.matcher = Matcher()
        self.frontKP,self.frontDES = self.matcher.sift_fet(templatefrontimg)
        self.backKP,self.backDES = self.matcher.sift_fet(templatebackimg)


    def run(self,im,threshold=0.7):
        KP,DES = self.matcher.sift_fet(im)
        frontmatches = self.matcher.flann.knnMatch(DES, self.frontDES, k=2)
        #frontmatches = self.matcher.bf.knnMatch(DES, self.frontDES, k=2)
        # print(frontmatches)
        frontgood = []
        for m, n in frontmatches:
            if m.distance < threshold * n.distance:
                frontgood.append(m)
        
        backmatches = self.matcher.flann.knnMatch(DES, self.backDES, k=2)
        #backmatches = self.matcher.bf.knnMatch(DES, self.backDES, k=2)
        # print(backmatches)
        backgood = []
        for m, n in backmatches:
            if m.distance < threshold * n.distance:
                backgood.append(m)
        
        if(len(frontgood)>len(backgood)):
            return 0
        else:
            return 1

   

if __name__ == '__main__':
    cls = IDCARDCLASS('/home/zhouzhao/Projects/IDCardClass/template/1.jpg','/home/zhouzhao/Projects/IDCardClass/template_back/1.jpg')
    imgpath = '/home/zhouzhao/Projects/pinganinvoice/test/身份证/img'
    for imgname in os.listdir(imgpath):
        print(imgname)
        image = cv2.imread(os.path.join(imgpath,imgname))
        # print(cls.run(image))
