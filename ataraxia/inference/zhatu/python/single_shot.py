#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np  
import sys
import os  
from argparse import ArgumentParser
if not 'demo/caffe/python' in sys.path:
    sys.path.insert(0,'demo/caffe/python') 

import caffe
from argparse import ArgumentParser
import time
import cv2
import random
import glob
from PIL import Image, ImageDraw, ImageFont
from os.path import basename

max_num_output = 4

ch11=[(0,1546),(960,0),(1380,0),(1380,2256)]
ch13=[(100,2256),(1015,0),(1370,0),(1800,2256)]
ch12=[(1370,2256),(1370,0),(1690,0),(2750,2256)]

dict_road = {"11":ch11, "12":ch12, "13":ch13}

def diff(p1,p2):
  return (p1[0]-p2[0], p1[1]-p2[1])

def cross_dot(p1, p2):
  return p1[0]*p2[1]-p2[0]*p1[1]


def line_eq(line_point1, line_point2):
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = line_point2[0] * line_point1[1] - line_point1[0] * line_point2[1]
    return A,B,C

def in_out(item, channel_vehicle):
    pts = dict_road[channel_vehicle]
    #A1,B1,C1 = line_eq(pts[0], pts[1])
    #A2,B2,C2 = line_eq(pts[2], pts[3])
    #x = int((item[2]+item[4])/2)
    x = int((item[4]-item[2])/3+item[2])
    y = int(item[5])
    #print('*****')
    #print(x)
    #print(y)
    #print(pts)
    d1 = cross_dot(diff(pts[0], pts[1]), diff((x,y),pts[1]))
    d2 = cross_dot(diff(pts[3], pts[2]), diff((x,y),pts[2]))
    #print(d1*d2<0)
    return (d1*d2 < 0)






def parser():
    parser = ArgumentParser('AtLab Demo!')
    parser.add_argument('--image',dest='im_path',help='Path to the image',
                        #default='zhatu2/right/right/right',type=str)
                        #default='zhatu2/exception/exception',type=str)
                        default='zhatu2/illegal/illegal/',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=0,type=int)
    parser.add_argument('--proto_refinedet',dest='prototxt_refinedet',help='refinedet prototxt',
                        default='models/refinedet/single_test_deploy.prototxt',type=str)
    parser.add_argument('--model_refinedet',dest='model_refinedet',help='refinedet caffemodel',
                        default='models/refinedet/coco_refinedet_resnet101_512x512_final.caffemodel',type=str)
    parser.add_argument('--proto_seresnet50v1',dest='prototxt_seresnet50v1',help='seresnet50v1 prototxt',
                        default='models/seresnet50v1/seresnet50v1.prototxt',type=str)
    parser.add_argument('--model_seresnet50v1',dest='model_seresnet50v1',help='seresnet50v1 caffemodel',
                        default='models/seresnet50v1/seresnet50v1.caffemodel',type=str)
    parser.add_argument('--proto_seresnet50v1_video',dest='prototxt_seresnet50v1_video',help='seresnet50v1 prototxt',
                        default='models/seresnet50v1/seresnet50v1.prototxt',type=str)
    parser.add_argument('--model_seresnet50v1_video',dest='model_seresnet50v1_video',help='seresnet50v1 caffemodel',
                        default='models/seresnet50v1/seresnet50v1_video.caffemodel',type=str)
    parser.add_argument('--proto_seresnet50v1_iflegal',dest='prototxt_seresnet50v1_iflegal',help='seresnet50v1 prototxt',
                        default='models/seresnet50v1/seresnet50v1.prototxt',type=str)
    parser.add_argument('--model_seresnet50v1_iflegal',dest='model_seresnet50v1_iflegal',help='seresnet50v1 caffemodel',
                        default='models/seresnet50v1/model_iflegal.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='zhatu2/right/output',type=str) 
    return parser.parse_args()


class Netmare:
    """
        Detect the max region of truck or bus in an image
    """
    def __init__(self, caffe_model, caffe_model2, caffe_model3, caffe_model4):
        self.caffe_model = caffe_model
        self.caffe_model2 = caffe_model2
        self.caffe_model3 = caffe_model3
        self.caffe_model4 = caffe_model4
    def preprocess(self, src):
        img = cv2.resize(src, (512,512))
        img = img.astype(np.float32, copy=False)
        img -= np.array([[[104.0, 117.0, 123.0]]])
        return img
    def postprocess(self, img, out):   
        h = img.shape[0]
        w = img.shape[1]
        box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
        cls = out['detection_out'][0,0,:,1]
        conf = out['detection_out'][0,0,:,2]
        return (box.astype(np.int32), conf, cls)
    def detect(self, img_org):
        img = self.preprocess(img_org)
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        self.caffe_model.blobs['data'].data[...] = img
        out = self.caffe_model.forward()  
        box, conf, cls = self.postprocess(img_org, out)
        output_detection = []
        # find max region id
        # flag false means no green slug car, 
        # flag true means return green slug car.
        for i in range(80):
            for j in range(len(box)):
                if int(cls[j]) == (i+1):
                    if i == 7 or i ==5: # truck or bus
                        if (box[j][2]-box[j][0] > 50) and (box[j][3]-box[j][1]>50) :
                            output_detection.append(box[j])
        return output_detection
    def VerifyGreen(self, img):
        img = cv2.resize(img, (225, 225))
        img = img.astype(np.float32, copy=True)
        img -= np.array([[[103.94,116.78,123.68]]])
        img = img * 0.017
        img = img.transpose((2, 0, 1))
        self.caffe_model2.blobs['data'].data[...] = img
        out = self.caffe_model2.forward()
        score = out['prob'][0]
        sort_pre = sorted(enumerate(score) ,key=lambda z:z[1])
        label_cls = [sort_pre[-j][0] for j in range(1,2)]
        #score_cls = [sort_pre[-j][1] for j in range(1,2)]
        if int(label_cls[0]) == 0:
            return True
        else:
            return False
    def classify(self, img_org, output_detection,video):
        classify_out = []
        for loop in range(max_num_output):
            classify_out.append([-1, 1, 0, 0, 0, 0]) # label, score, box(4)
        count = 0
        for element in output_detection:
            img = img_org[max(0,element[1]):element[3],max(0, element[0]):element[2]] 
            if video:
                img = cv2.resize(img, (225, 225))
                img = img.astype(np.float32, copy=True)
                img -= np.array([[[103.94,116.78,123.68]]])
                img = img * 0.017
                img = img.transpose((2, 0, 1))
                self.caffe_model3.blobs['data'].data[...] = img
                out = self.caffe_model3.forward()
            else:
                if not self.VerifyGreen(img):
                    continue
                rows, cols, ch = img.shape
                img = img[0:int(rows/6), 0:cols]
                img = cv2.resize(img, (225, 225))
                img = img.astype(np.float32, copy=True)
                img -= np.array([[[103.94,116.78,123.68]]])
                img = img * 0.017
                img = img.transpose((2, 0, 1))
                self.caffe_model4.blobs['data'].data[...] = img
                out = self.caffe_model4.forward()
            score = out['prob'][0]
            sort_pre = sorted(enumerate(score) ,key=lambda z:z[1])
            label_cls = [sort_pre[-j][0] for j in range(1,2)]
            score_cls = [sort_pre[-j][1] for j in range(1,2)]
            if video:
                if count < max_num_output and int(label_cls[0]) == 0: 
                    classify_out[count][0] = int(label_cls[0])
                    classify_out[count][1] = float(score_cls[0])
                    classify_out[count][2] = int(element[0])
                    classify_out[count][3] = int(element[1])
                    classify_out[count][4] = int(element[2])
                    classify_out[count][5] = int(element[3])
                    count += 1
            else:
                if count < max_num_output: 
                    classify_out[count][0] = int(label_cls[0])
                    classify_out[count][1] = float(score_cls[0])
                    classify_out[count][2] = int(element[0])
                    classify_out[count][3] = int(element[1])
                    classify_out[count][4] = int(element[2])
                    classify_out[count][5] = int(element[3])
                    count += 1
        return classify_out
    def process(self,img_org,video,name):
        output_detection = self.detect(img_org)
        width_image = int(img_org.shape[1])
        classify_out = self.classify(img_org, output_detection, video)
        if len(name) < 55:
            for i in range(len(classify_out)):
                classify_out[i][0] = -1
        channel_vehicle = name[14:16]
        for element in classify_out:
            if int(element[0]) == 0: # 1 means illegal green car
                element[0] = 1
            elif int(element[0]) == 1: # 0 means normal green car
                element[0] = 0
        if not video : # zhua pai return only one green car 
            id = 0
            max_contour = 0
            for i in range(len(classify_out)):
                #print('########')
                #print(classify_out[i][0])
                classify_out[i][2] = max(0,classify_out[i][2])
                classify_out[i][3] = max(0,classify_out[i][3])
                area = (classify_out[i][4]-classify_out[i][2])*(classify_out[i][5]-classify_out[i][3])
                if area < 500000:
                    classify_out[i][0] = -1 # remove green car which is at side, not the main object
            if width_image > 2000 :
                for i in range(len(classify_out)):
                    if not in_out(classify_out[i],channel_vehicle):
                        classify_out[i][0] = -1
                        #print('$$$$$')
                        #print(classify_out[i][0])
            for i in range(len(classify_out)):
                if  area > max_contour and not classify_out[i][0]<0:
                    max_contour = area
                    id = i
            for i in range(len(classify_out)):
                if not i == id :
                    classify_out[i][0] = -1
        return classify_out




args = parser()
caffe.set_mode_gpu()
caffe.set_device(1)

#net_refinedet = caffe.Net(str(configs['model_files']["prototxt_refinedet"]), str(configs['model_files']["model_refinedet"]), caffe.TEST)
#net_seresnet = caffe.Net(str(configs['model_files']["proto_seresnet50v1"]), str(configs['model_files']["model_seresnet50v1"]), caffe.TEST)
#net_iflegal = caffe.Net(str(configs['model_files']["proto_seresnet50v1_iflegal"]), str(configs['model_files']["model_seresnet50v1_iflegal"]), caffe.TEST)
#net_videonet = caffe.Net(str(configs['model_files']["proto_seresnet50v1"]), str(configs['model_files']["model_videonet50v1"]), caffe.TEST)

net_refinedet = caffe.Net( args.prototxt_refinedet, args.model_refinedet, caffe.TEST)
net_seresnet = caffe.Net(args.prototxt_seresnet50v1, args.model_seresnet50v1, caffe.TEST)
net_videonet = caffe.Net( args.prototxt_seresnet50v1_video, args.model_seresnet50v1_video, caffe.TEST)
net_iflegal = caffe.Net( args.prototxt_seresnet50v1_iflegal, args.model_seresnet50v1_iflegal, caffe.TEST)
net_iflegal2 = caffe.Net( args.prototxt_seresnet50v1_iflegal, args.model_seresnet50v1_iflegal, caffe.TEST)


net_refinedet.name = 'refinedet_resnet101_512x512'
net_seresnet.name = 'seresnet50v1_225x225'
net_seresnet.name = 'seresnet50v1_225x225_iflegal'
net_videonet.name = 'videonet_225x225'
netmare = Netmare(net_refinedet, net_seresnet, net_videonet, net_iflegal)  
for image in os.listdir(args.im_path):
    #if image.find('_01_'):
    if True:
        im_path = os.path.join(args.im_path,image)
        #print(im_path)
        img = cv2.imread(im_path)
        if img is not None:
            #print(im_path)
            classify_out = netmare.process(img,False,image)
            for k in range(len(classify_out)):
                #print(classify_out[k][0])
                if True:
                #if classify_out[k][0] == 1:
                    print(classify_out[k][0])
                    print(im_path)
                    cv2.imwrite(os.path.join(args.out_path,'full',(image)), img)
                    img_crop = img[max(0,classify_out[k][3]):classify_out[k][5],max(0, classify_out[k][2]):classify_out[k][4]]
                    cv2.imwrite(os.path.join(args.out_path,'part',(image).split()[0]+str(k)+'.jpg'), img_crop)
                    rows, cols, ch = img_crop.shape
                    crop_head = img_crop[0:int(rows/6), 0:cols]
                    cv2.imwrite(os.path.join(args.out_path,'crop',(image).split()[0]+str(k)+'.jpg'),crop_head)




