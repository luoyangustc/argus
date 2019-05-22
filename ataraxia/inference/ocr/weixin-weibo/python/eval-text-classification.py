# -*- coding: utf-8 -*-
#coding=utf-8
import numpy as np
import cv2
import caffe
import argparse
import json
import traceback

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
        monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image

def init_models(use_divice, net_def_file, model_file):
    if use_divice == "GPU":
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    # initialize the cls model
    cls_mod = caffe.Net(str(net_def_file),str(model_file),caffe.TEST)
    return cls_mod

def cls_process(net_cls, img):
    img = cv2.resize(img, (225, 225))
    img = img.astype(np.float32, copy=True)
    img -= np.array([[[103.94,116.78,123.68]]])
    img = img * 0.017
    img = img.transpose((2, 0, 1))
    net_cls.blobs['data'].data[...] = img
    out = net_cls.forward()
    score = out['prob'][0]
    sort_pre = sorted(enumerate(score) ,key=lambda z:z[1])
    label_cls = [sort_pre[-j][0] for j in range(1,2)]
    score_cls = [sort_pre[-j][1] for j in range(1,2)]
    return label_cls, score_cls

def save_json_in_text(json_dict=None, text_path=None):
    with open(text_path,'w') as f:
        json_result = json.dumps(json_dict, ensure_ascii=False)
        print json_result
        f.write(json_result)
        f.flush()
        f.close()
    pass

def process_image_fun(net_cls=None, origimg=None, threshold=0.6):
    cls_dict = {28: 'blog', 29: 'wechat', 30: 'other-text'}
    CTX.logger.info("np.shape: %s", np.shape(origimg))
    if np.shape(origimg) != ():
        label_cls, score_cls = cls_process(net_cls, origimg)
        img_type = ""
        cla_index = int(label_cls[0])
        if float(score_cls[0]) > threshold and cla_index in cls_dict:
            img_type = cls_dict.get(cla_index)
        else:
            img_type = "others"

    confidences = [{
        "index": int(cla_index),
        "class": str(img_type),
        "score": float(score_cls[0]),
    }]
    return {"code": 0, "message": "", "result": {"confidences": confidences}}

def parse_args():
    parser = argparse.ArgumentParser(description='AtLab Label Image!',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    args = parser.parse_args()
    return args

@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    text_classify = init_models(configs['use_device'].upper(),
                                configs['model_files']['deploy.prototxt'], 
                                configs['model_files']['text-classification-v0.2-t3.caffemodel'])
    return {"text_classify": text_classify, "batch_size": configs['batch_size'], "threshold": configs['custom_params']['threshold']}, 0, ''

@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''

@net_inference_handler
def net_inference(model, reqs):
    text_classify = model['text_classify']
    batch_size = model['batch_size']
    threshold = model['threshold']
    CTX.logger.info("inference begin ...")
    CTX.logger.info("requests: %s", reqs)

    try:
        ret = []
        for i in range(len(reqs)):
            img = load_image(reqs[i]["data"]["uri"], body=reqs[i]['data']['body'])
            resp = process_image_fun(text_classify, img, threshold)
            ret.append(resp)

    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)

    return ret, 0, ''


if __name__ == "__main__":
    cls_mod = init_models("GPU","","")
    image_path = "http://p24v9nypo.bkt.clouddn.com/005Zu0d2ly1fliuvd70yyj30hs1vrajd.jpg"
    img = cv2.imread(image_path)
    res = process_image_fun(cls_mod,img,0.6)
    print(res)