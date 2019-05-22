#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import numpy as np
import struct
import random
from argparse import ArgumentParser

def meantype(s):
    try:
        return map(float, s.split(','))
    except:
        raise argparse.ArgumentTypeError("value must be b,g,r")

def parser():
    parser = ArgumentParser('AtLab Tensorrt Int8 Calibrate!')
    parser.add_argument('--ImgPath',dest='path',help='Path to the image',
                        default='build/imagenetval',type=str)
    parser.add_argument('--ImgName',dest='name',help='Database name',
                        default='Resnet50',type=str)
    parser.add_argument('--Number',dest='N',help='Image number in a batch',
                        default=100,type=int)
    parser.add_argument('--Width',dest='W',help='Net input width',
                        default=224,type=int)
    parser.add_argument('--Height',dest='H',help='Net input height',
                        default=224,type=int)
    parser.add_argument('--Mean',dest='mean',help='Net input mean [b,g,r]',
                        default=[128, 128, 128],type=meantype)
    parser.add_argument('--Var',dest='var',help='Net input variance',
                        default=1.0,type=float)
    return parser.parse_args()

def fileList(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':  
                L.append(file)
            if os.path.splitext(file)[1] == '.JPG':  
                L.append(file)  
            if os.path.splitext(file)[1] == '.png':  
                L.append(file)  
            if os.path.splitext(file)[1] == '.PNG':  
                L.append(file)  
            if os.path.splitext(file)[1] == '.jpeg':  
                L.append(file)  
            if os.path.splitext(file)[1] == '.JPEG':  
                L.append(file)    
    return L 

def generateBatchFile(Path, name, height, width, mean, var, N):
    imageCount = 0
    fileCount = 0
    if not os.path.exists(os.path.join("build","batches")):
        os.mkdir(os.path.join("build","batches"))
    if not os.path.exists(os.path.join("build/batches",name)):
        os.mkdir(os.path.join("build/batches",name))
    bathPath = os.path.join("build/batches",name);
    imglist = fileList(Path)
    labelList = [] 
    for imgFile in imglist:
        print imageCount
        if imageCount % N == 0:
            f = open(os.path.join(bathPath, "batch" + str(fileCount)),"wb")
            f.write(struct.pack("iiii", N, 3, height, width))
            fileCount += 1
        #image
        image = cv2.imread(os.path.join(Path, imgFile))
        image = cv2.resize(image,(width,height))
        image = np.array(image, dtype = 'float32')
        image = (image - mean) * var
        image = image.transpose(2,0,1)
        image = np.array(image, dtype = 'float32')
        f.write(image.tobytes())
        #label
        imageName = imgFile.split(".")[0]
        labelFile =  imageName + ".txt"
        labelObject = open(os.path.join(Path,labelFile))
        for line in labelObject:
 	    labelList.append(float(line))
        imageCount += 1 
        
        if  imageCount % N == 0:
            labelArray = np.array(labelList,dtype = 'float32')
            f.write(labelArray.tobytes())
            labelList = []
            f.close()
    print("一共处理图片{}张，共产生{}个batch".format(imageCount, fileCount))
    if imageCount % N != 0:
        os.remove(os.path.join(bathPath, "batch" + str(fileCount - 1)))
        print("最后{}张图片不足{}张，不能构建batch,已自动删除batch{}".format(imageCount % N , N, fileCount - 1))
    f.close()

if __name__ == "__main__":
    args = parser()
    print args
    #batch generate
    generateBatchFile(args.path, args.name, args.H, args.W, args.mean, args.var, args.N)
            
