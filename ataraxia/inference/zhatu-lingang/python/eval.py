
import numpy as np  
import sys
import os  
import caffe
import time
import cv2
import random
import glob
import traceback
from PIL import Image, ImageDraw, ImageFont
import json
from argparse import ArgumentParser

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
        monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image

max_num_output = 4


def diff(p1,p2):
  return (p1[0]-p2[0], p1[1]-p2[1])

def cross_dot(p1, p2):
  return p1[0]*p2[1]-p2[0]*p1[1]


def line_eq(line_point1, line_point2):
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = line_point2[0] * line_point1[1] - line_point1[0] * line_point2[1]
    return A,B,C

def in_out(item, pts):
    #A1,B1,C1 = line_eq(pts[0], pts[1])
    #A2,B2,C2 = line_eq(pts[2], pts[3])
    x = int((item[4]-item[2])/3+item[2])
    y = int(item[5])
    d1 = cross_dot(diff(pts[0], pts[1]), diff((x,y),pts[1]))
    d2 = cross_dot(diff(pts[3], pts[2]), diff((x,y),pts[2]))
    return (d1*d2 < 0)



def parser():
    parser = ArgumentParser('AtLab Demo!')
    parser.add_argument('--image',dest='im_path',help='Path to the image',
                        default='test/input/video',type=str)
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
                        default='test/output/',type=str) 
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

    def process(self, img_org, video, name, pts):
        output_detection = self.detect(img_org)
        width_image = int(img_org.shape[1])
        classify_out = self.classify(img_org, output_detection, video)
        #if len(name) < 55:
        #    for i in range(len(classify_out)):
        #        classify_out[i][0] = -1
        for element in classify_out:
            if int(element[0]) == 0: # 1 means illegal green car
                element[0] = 1
            elif int(element[0]) == 1: # 0 means normal green car
                element[0] = 0
        if not video: # zhua pai return only one green car 
            id = 0
            max_contour = 0
            for i in range(len(classify_out)):
                classify_out[i][2] = max(0,classify_out[i][2])
                classify_out[i][3] = max(0,classify_out[i][3])
                area = (classify_out[i][4]-classify_out[i][2])*(classify_out[i][5]-classify_out[i][3])
                #if area < 500000:
                #if area < 100000:
                #    classify_out[i][0] = -1 # remove green car which is at side, not the main object
            #if width_image > 2000 : # resolution > 2000
            #if width_image > 1000 : # width > 1000
            #    for i in range(len(classify_out)):
            #        if not in_out(classify_out[i], pts):  # to check if car in right channel
            #            classify_out[i][0] = -1
            for i in range(len(classify_out)): # to get unique green car
                #if  area > max_contour and not classify_out[i][0]<0:
                if not classify_out[i][0]<0:
                    max_contour = area
                    id = i
            for i in range(len(classify_out)):
                if not i == id :
                    classify_out[i][0] = -1
        return classify_out


@create_net_handler
def create_net(configs):
    args = parser()
    CTX.logger.info("load configs: %s", configs)
    caffe.set_mode_gpu()
    net_refinedet = caffe.Net(str(configs['model_files']["prototxt_refinedet"]), str(configs['model_files']["model_refinedet"]), caffe.TEST)
    net_seresnet = caffe.Net(str(configs['model_files']["proto_seresnet50v1"]), str(configs['model_files']["model_seresnet50v1"]), caffe.TEST)
    net_iflegal = caffe.Net(str(configs['model_files']["proto_seresnet50v1_iflegal"]), str(configs['model_files']["model_seresnet50v1_iflegal"]), caffe.TEST)
    net_videonet = caffe.Net(str(configs['model_files']["proto_seresnet50v1"]), str(configs['model_files']["model_videonet50v1"]), caffe.TEST)
    net_refinedet.name = 'refinedet_resnet101_512x512'
    net_seresnet.name = 'seresnet50v1_225x225'
    net_seresnet.name = 'seresnet50v1_225x225_iflegal'
    net_videonet.name = 'videonet_225x225'
    netmare = Netmare(net_refinedet, net_seresnet, net_videonet, net_iflegal)
    return {"net": netmare, "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    netmare = model['net']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")
    try:
        imges_with_type = pre_eval(batch_size, reqs)
        output = eval(netmare, imges_with_type)
        ret = post_eval(output, reqs)
    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)
    return ret, 0, ''


def pre_eval(batch_size, reqs):
    cur_batchsize = len(reqs)
    if cur_batchsize > batch_size:
        raise ErrorOutOfBatchSize(batch_size)  
    ret = []
    _t1 = time.time()
    for i in range(cur_batchsize):
        img = load_image(reqs[i]["data"]["uri"], body=reqs[i]['data']['body'])
        img_type = reqs[i]["data"]["attribute"].get("image_type", 0)
        img_name = reqs[i]["data"]["attribute"].get("name", "")
        video = reqs[i]["data"]["attribute"].get("video", False)
        lane_pts = reqs[i]["data"]["attribute"].get("lane_pts")
        ret.append((img, img_type, video, img_name, lane_pts))
    _t2 = time.time()
    CTX.logger.info("load: %f", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)
    return ret


def post_eval(output, reqs=None):
    resps = []
    cur_batchsize = len(output)
    _t1 = time.time()
    for i in xrange(cur_batchsize):
        result = dump_result(output[i])
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(netmare, imges_with_type):
    output = []
    _t1 = time.time()
    for i in range(len(imges_with_type)):
        classify_out = netmare.process(imges_with_type[i][0], imges_with_type[i][2], imges_with_type[i][3], imges_with_type[i][4])
        output.append(classify_out) # label 0 means illegal slug car, label 1 means other legal slug car
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output

def dump_result(classify_out):
    ifslugcar_result = dict()
    ifslugcar_result["detections"] = list()
    for loop in range(max_num_output):
        if classify_out[loop][0] < 0:
            continue
        result = dict()
        # for label, 0 is target car, 1 is normal car, -1 means no car
        result['label'] = classify_out[loop][0] 
        result['score'] = classify_out[loop][1]
        result['pts'] = list()
        result['pts'].append(classify_out[loop][2])
        result['pts'].append(classify_out[loop][3])
        result['pts'].append(classify_out[loop][4])
        result['pts'].append(classify_out[loop][5])
        ifslugcar_result["detections"].append(result)
    return ifslugcar_result
