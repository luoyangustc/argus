import torch
import yaml
import config
import numpy as np
from . import keys,crann
from src.basemodel.basemodel import BaseModel
from .quadObject import quadObject
from .opImage import genRotateAroundMatrix, rotateImageByMatrix
import base64
import cv2
import json
from skimage import io
import logging


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
        self.model.load_state_dict(torch.load(modelpath)['state_dict'])
        if(config.USE_GPU):
            self.model.half()
    
    def cutimage(self,dat):
        dat = json.loads(dat)
        img = io.imread(dat['imgurl'])
        logging.critical('img:'+str(img.shape))
        img = cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)
        # logging.critical(img.shape)
        bboxes = dat['bboxes']
        num = 0
        showimg = img.copy()
        imglist = []
        # for box in bboxes:
        #     box = np.array(box)
        #     for i in range(4):
        #         cv2.line(showimg,tuple(box[i]),tuple(box[(i+1)%4]),(0,0,255),3)
        #     box_obj = quadObject(box)
        #     logging.critical('angle:'+str(box_obj.angle_of_center_line))
        #     M = box_obj.genRotateAroundMatrix(box_obj.center_point, box_obj.angle_of_center_line)
        #     box_obj1 = box_obj.rotateByMatrix(M)

        #     new_box = box_obj1.quad
        #     x = new_box[:, 0]
        #     y = new_box[:, 1]

        #     rotated_img = rotateImageByMatrix(img, M)

        #     left = np.min(new_box[:, 0])
        #     right = np.max(new_box[:, 0])
        #     bottom = np.min(new_box[:, 1])
        #     top = np.max(new_box[:, 1])


        #     part_img = rotated_img[bottom:top, left:right]

        #     if 0 in part_img.shape:
        #         part_img = np.ones((32,32,3), dtype=np.uint8) * 255

        #     cv2.imwrite('cut_img'+str(num)+'.jpg',part_img)
        #     num+=1
        for box in bboxes:
            box = np.array(box)
            
            # for i in range(4):
            #     cv2.line(showimg,tuple(box[i]),tuple(box[(i+1)%4]),(0,0,255),3)

            x0 = np.min(box[:, 0])
            y0 = np.min(box[:, 1])
            x1 = np.max(box[:, 0])
            y1 = np.max(box[:, 1])

            box = np.stack([box[:, 0]-x0, box[:, 1]-y0], axis=1)
            sub_img = img[y0:y1+1, x0:x1+1].copy()

            if np.min(sub_img.shape) == 0:
                part_img = np.ones((32,32,3), dtype=np.uint8) * 255
            else:
                box_obj = quadObject(box)

                M = box_obj.genRotateAroundMatrix(box_obj.center_point, box_obj.angle_of_center_line)
                box_obj1 = box_obj.rotateByMatrix(M)

                new_box = box_obj1.quad
                # x = new_box[:, 0]
                # y = new_box[:, 1]

                # logging.critical(M.shape)
                # logging.critical(sub_img.shape)

                rotated_img = rotateImageByMatrix(sub_img, M)

                left = np.min(new_box[:, 0])
                right = np.max(new_box[:, 0])
                bottom = np.min(new_box[:, 1])
                top = np.max(new_box[:, 1])
                # logging.critical(str(top+1-bottom)+'/'+str(right+1-left))
                # logging.critical(rotated_img.shape)
                part_img = rotated_img[bottom:top+1, left:right+1]

                if 0 in part_img.shape:
                    part_img = np.ones((32,32,3), dtype=np.uint8) * 255
            # logging.critical(part_img.shape)
            # cv2.imwrite('cut_img'+str(num)+'.jpg',part_img)
            num+=1
            imglist.append(255 - cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY))
        # cv2.imwrite('img.jpg',showimg)
        return imglist