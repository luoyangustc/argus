#coding:UTF-8
import torch
import yaml
import config
import numpy as np
from . import keys,crann
from evals.src.basemodel.basemodel import BaseModel
from .quadObject import quadObject
from .opImage import genRotateAroundMatrix, rotateImageByMatrix
import base64
import cv2
import json
import logging

SAVEIMG = False
class CrannRecModel(BaseModel):
    """docstring for upperCase"""
    def __init__(self, modelpath, config_yaml):
        super(CrannRecModel, self).__init__()
        self.alphabet = keys.alphabet
        f = open(config_yaml)
        opt = yaml.load(f)
        opt['N_GPU'] = 1
        opt['RNN']['multi_gpu'] = False
        # print(opt)
        self.model = crann.CRANN(opt, len(self.alphabet)+1)
        if(config.USE_GPU):
            self.model.cuda()
        # self.model.half()
        self.num = 0
        self.model.load_state_dict(torch.load(modelpath)['state_dict'])
        if(config.USE_GPU):
            self.model.half()
    

    def cutimagezz(self,img,bboxes):
        showimg = img.copy()
        imglist = []
        for box in bboxes:
            box = np.array(box, dtype=np.int32)

            L = np.min(box[:,0])
            T = np.min(box[:,1])
            R = np.max(box[:,0])
            B = np.max(box[:,1])
            if R<=L or B<=T:
                part_img = np.ones((32,32,3), dtype=np.uint8) * 255
            else:
                part_img = showimg[T:B,L:R,:]
            # if SAVEIMG:
            #     cv2.imwrite('./disp/'+str(self.num)+'.jpg',part_img)    
            #     self.num+=1

            imglist.append(255 - cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY))
        # cv2.imwrite('img.jpg',showimg)
        return imglist