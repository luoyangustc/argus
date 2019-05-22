# -*- coding:utf-8 -*-
"""
terror cascade classification
the script is used as inference code
"""
import sys
import os
# because in zxt-det-t2 pod
#sys.path.insert(0, '/workspace/data/wbb/caffe_new/python')
import numpy as np
import cv2
import caffe
import argparse
import json
from cfg import Config
import urllib
import time
cur_path = os.path.abspath(os.path.dirname(__file__))
#cur_path = cur_path[:cur_path.rfind('/')]

def init_models():
    if Config.PLATFORM == "GPU":
        caffe.set_mode_gpu()
        caffe.set_device(Config.TEST_GPU_ID)
    else:
        caffe.set_mode_cpu()
    # initialize the cls model
    cls_mod = caffe.Net(Config.CLS_NET_DEF_FILE,Config.CLS_MODEL_PATH,caffe.TEST)
    return cls_mod

def txt_to_dict():
    label_index = np.loadtxt(Config.CLS_LABEL_INDEX,str,delimiter='\n')
    cls_dict = {}
    for index in label_index:
        cls_dict.update({index.split(',')[0]:index.split(',')[1]})
    return cls_dict

def classify_flow(image_name=None,label_cls=None,index=None,score=None):
    flow_dict = {
        "class":label_cls,
        "index":index,
        "score":score
    }
    if args.urllist is not None:
        # just get image name from iamge url
        image_name = image_name.split('/')[-1]
    # image local path  ,save absolute image path and name
    flow =  "%s\t%s"%(image_name,json.dumps(flow_dict, ensure_ascii=False))
    return flow

def center_crop(img, crop_size):
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size]

def cls_process(net_cls, img):
    img = img.astype(np.float32, copy=True)
    img = cv2.resize(img, (256, 256))
    img -= np.array([[[103.94,116.78,123.68]]])
    img = img * 0.017
    img = center_crop(img, 225)
    img = img.transpose((2, 0, 1))
    net_cls.blobs['data'].data[...] = img
    out = net_cls.forward()
    score = out['prob'][0]
    sort_pre = sorted(enumerate(score) ,key=lambda z:z[1])
    label_cls = [sort_pre[-j][0] for j in range(1,2)]
    score_cls = [sort_pre[-j][1] for j in range(1,2)]
    return label_cls, score_cls

def process_image_fun(net_cls=None, image_list=None,cls_dict=None):
    write_list=[]
    for image_path in image_list:
        if args.urllist is not None:
            data = urllib.urlopen(image_path).read()
            nparr = np.fromstring(data,np.uint8)
            origimg = cv2.imdecode(nparr,1)
        else:
            origimg = cv2.imread(image_path)
        if np.shape(origimg) != ():
            starttime = time.time()
            label_cls, score_cls = cls_process(net_cls, origimg)
            endtime = time.time()
            print ("%s speed: %s"%(image_path,str(endtime-starttime)))
            label_cls = str(label_cls[0])
            score_cls = float(score_cls[0])
            write_line = classify_flow(image_name=image_path,label_cls=cls_dict.get(label_cls),index=int(label_cls),score=score_cls)
            write_list.append(write_line)
    timeFileFlag = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    with open(os.path.join(cur_path,'result-'+str(timeFileFlag)+'.txt'),'w') as f:
        for w in write_list:
            f.write(w)
            f.write('\n')
            f.flush()
def parse_args():
    parser = argparse.ArgumentParser(description='AtLab Label Image!',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    parser.add_argument('--urllist', help='input image url list', default=None, type=str)
    parser.add_argument('--inputImagePath', help='input image local baes path', default=None, type=str)
    args = parser.parse_args()
    return args
def checkFileIsImags(filePath):
    if ('JPEG' in filePath.upper()) or ('JPG' in filePath.upper()) \
            or ('PNG' in filePath.upper()) or ('BMP' in filePath.upper()) \
            or ('GIF' in filePath.upper()):
        return True
    return False
    pass
args = parse_args()
if __name__ == "__main__":
    cls_mod = init_models()
    cls_dict = txt_to_dict()
    need_process_images_list=[]
    if args.image is not None:
        if checkFileIsImags(args.image):
            need_process_images_list.append(args.image)
        process_image_fun(net_cls=cls_mod,image_list=need_process_images_list,cls_dict=cls_dict)
    elif args.imagelist is not None:
        with open(args.imagelist,'r') as f:
            for line in f.readlines():
                if checkFileIsImags(line.strip()):
                    need_process_images_list.append(line.strip())
        process_image_fun(net_cls=cls_mod,image_list=need_process_images_list,cls_dict=cls_dict)
    elif args.urllist is not None:
        with open(args.urllist,'r') as f:
            for line in f.readlines():
                if len(line.strip()) != 0:
                    need_process_images_list.append(line.strip())
        process_image_fun(net_cls=cls_mod,image_list=need_process_images_list,cls_dict=cls_dict)
        pass
    elif args.inputImagePath is not None:
        for parent, dirnames, filenames in os.walk(args.inputImagePath):
            for filename in filenames:
                imagePath = os.path.join(parent, filename)
                if checkFileIsImags(imagePath):
                    need_process_images_list.append(imagePath)
        process_image_fun(net_cls=cls_mod,image_list=need_process_images_list,cls_dict=cls_dict)
