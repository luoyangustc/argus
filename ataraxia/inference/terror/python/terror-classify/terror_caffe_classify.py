#coding=utf-8

import os
import sys
import time
import traceback
import numpy as np
import csv
import caffe
import cv2
import json
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format

from collections import OrderedDict

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


def change_deploy(deploy_file=None, input_data_batch_size=None):
    net = caffe_pb2.NetParameter()
    with open(deploy_file, 'r') as f:
        text_format.Merge(f.read(), net)
    data_layer_index = 0
    data_layer = net.layer[data_layer_index]
    data_layer.input_param.shape[0].dim[0] = input_data_batch_size
    with open(deploy_file, 'w') as f:
        f.write(str(net))


def get_labellist(labelmap):
    '''
        get label list
        Return
        ----------
        labelList: label index and class name, list
    '''
    labelList = {}
    with open(labelmap, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            index = int(line[0])
            cls_name = line[1]
            labelList[index] = cls_name
    return labelList


@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    caffe.set_mode_gpu()
    deploy = str(configs['model_files']["deploy.prototxt"])
    weight = str(configs['model_files']["weight.caffemodel"])
    labelmap = str(configs['model_files']["labels.csv"])
    if "batch_size" in configs:
        batch_size = configs['batch_size']
    else:
        batch_size = 1
    # change deploy file input data dim
    change_deploy(deploy_file=deploy, input_data_batch_size=batch_size)
    net = caffe.Net(deploy, weight, caffe.TEST)
    threshold = 0.9
    if 'custom_params' in configs:
        custom_params = configs['custom_params']
        if 'threshold' in custom_params:
            threshold = custom_params['threshold']
    return {"net": net, "threshold": threshold, "batch_size": batch_size, "label_list": get_labellist(labelmap)}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):

    net = model["net"]
    threshold = model["threshold"]
    batch_size = model['batch_size']
    label_list = model["label_list"]
    CTX.logger.info("inference begin ...")
    try:
        images = pre_eval(net, batch_size, reqs)
        output = eval(net, images)
        cur_batchsize = len(images)
        ret = post_eval(output, threshold, cur_batchsize, label_list)
    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)
    return ret, 0, ''


def center_crop(img, crop_size):
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy:yy + crop_size, xx: xx + crop_size]


def preProcessImage(img=None):
    img = img.astype(np.float32, copy=True)
    img = cv2.resize(img, (256, 256))
    # img = img.astype(np.float32, copy=True)
    img -= np.array([[[103.94, 116.78, 123.68]]])
    img = img * 0.017
    img = center_crop(img, 225)
    img = img.transpose((2, 0, 1))
    return img


def pre_eval(net, batch_size, reqs):
    '''
        prepare net forward data
        Parameters
        ----------
        net: net created by net_init
        reqs: parsed reqs from net_inference
        reqid: reqid from net_inference
        Return
        ----------
        code: error code, int
        message: error message, string
    '''
    cur_batchsize = len(reqs)
    CTX.logger.info("cur_batchsize: %d\n", cur_batchsize)
    if cur_batchsize > batch_size:
        for i in range(cur_batchsize):
            raise ErrorOutOfBatchSize(batch_size)
    images = []
    _t1 = time.time()
    for i in range(cur_batchsize):
        data = reqs[i]
        img = load_image(data["data"]["uri"], body=data['data']['body'])
        if img is None:
            CTX.logger.info("input data is none : %s\n", data)
            raise ErrorBase(400, "image data is None ")
        if img.ndim != 3:
            raise ErrorBase(400, "image ndim is " +
                            str(img.ndim) + ", should be 3")
        images.append(preProcessImage(img))
    _t2 = time.time()
    CTX.logger.info("read image and transform: %f\n", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)
    return images


def post_eval(output, threshold, cur_batchsize, label_list):
    '''
        parse net output, as numpy.mdarray, to EvalResponse
        Parameters
        ----------
        net: net created by net_init
        output: list of tuple(score, boxes)
        reqs: parsed reqs from net_inference
        reqid: reqid from net_inference
        label_list: label list of labels.csv
        Return
        ----------
        resps: list of EvalResponse{
            "code": <code|int>,
            "message": <error message|str>,
            "result": <eval result|object>,
            "result_file": <eval result file path|string>
        }
    '''
    resps = []
    _t1 = time.time()

    for index, output_prob in enumerate(output['prob']):
        if index >= cur_batchsize:
            break
        output_prob = np.squeeze(output['prob'][index])

        result = {}
        confidences = []
        index = int(output_prob.argsort()[-1])
        class_name = str(label_list[index])
        score = float(output_prob[index])
        if score < threshold and class_name != 'normal':
            index = -1
        confidence = {
            "index": index,
            "class": class_name,
            "score": score
        }
        if class_name != "":
            confidences.append(confidence)
        result["confidences"] = confidences
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f\n", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(net, images):
    '''
        eval forward inference
        Return
        ---------
        output: network numpy.mdarray
    '''
    _t1 = time.time()
    for index, i_data in enumerate(images):
        net.blobs['data'].data[index] = i_data
    _t2 = time.time()
    CTX.logger.info("load image to net: %f\n", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)

    _t1 = time.time()
    output = net.forward()
    _t2 = time.time()
    CTX.logger.info("forward: %f\n", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)

    if 'prob' not in output or len(output['prob']) < len(images):
        raise ErrorForwardInference()
    return output
