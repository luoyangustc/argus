# -*- coding: utf-8 -*-
"""
    note : 201810 ,debug : error image in a batch
"""
import os
import sys
import time
import traceback
import numpy as np
import csv
import caffe
import cv2
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format
import hashlib
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

@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    if 'use_device' in configs and configs['use_device'] == "CPU":
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
    deploy = str(configs['model_files']["deploy.prototxt"])
    weight = str(configs['model_files']["weight.caffemodel"])
    labelmap = str(configs['model_files']["labels.csv"])
    batch_size = 1
    if "batch_size" in configs:
        batch_size = configs['batch_size']
    # change deploy file input data dim
    change_deploy(deploy_file=deploy, input_data_batch_size=batch_size)
    net = caffe.Net(deploy, weight, caffe.TEST)
    if 'custom_params' in configs:
        custom_params = configs['custom_params']
        if 'thresholds' in custom_params:
            thresholds = custom_params['thresholds']
        else:
            thresholds = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]
    return {"net": net, "thresholds": thresholds, "batch_size": batch_size, "label_list": get_labellist(labelmap)}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):

    net = model["net"]
    thresholds = model["thresholds"]
    batch_size = model['batch_size']
    label_list = model["label_list"]
    CTX.logger.info("inference begin ...")
    try:
        imageReadInfoDict,imageIdMapDict,images = pre_eval(net, batch_size, reqs)
        output = eval(net, images)
        ret = post_eval(imageIdMapDict, imageReadInfoDict, output, thresholds, label_list)
    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)
    return ret, 0, ''


def preProcessImage(oriImage=None):
    img = cv2.resize(oriImage, (320, 320))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    img = img.transpose((2, 0, 1))
    return img


def pre_eval(net, batch_size, reqs):
    '''
        prepare net forward data
    '''
    imageReadInfoDict = dict()
    images = []
    netinfeImageIdToinputImageId = dict() # net infe image id map to input image id
    cur_batchsize = len(reqs)
    CTX.logger.info("cur_batchsize: %d\n", cur_batchsize)
    if cur_batchsize > batch_size:
        raise ErrorOutOfBatchSize(batch_size)
    # image_shape_list_h_w = []
    # images = []
    _t1 = time.time()
    normalImageIndex = 0
    for i in range(cur_batchsize):
        data = reqs[i]
        infoOfImage = dict()
        img = None
        try:
            # load image error
            img = load_image(data["data"]["uri"],body=data['data']['body'])
            if img is None:
                CTX.logger.info("input data is none : %s\n", data)
                infoOfImage['errorInfo'] = "image data is None"
                infoOfImage['errorCode'] = 400
                infoOfImage['flag'] = 1
            elif img.ndim != 3:
                CTX.logger.info("image ndim is " +
                                str(img.ndim) + ", should be 3\n")
                infoOfImage['errorInfo'] = "image ndim is " + \
                    str(img.ndim) + ", should be 3"
                infoOfImage['errorCode'] = 400
                infoOfImage['flag'] = 1
        except ErrorBase as e:
            CTX.logger.info("image of index : %d,preProcess error: %s\n", i,str(e))
            infoOfImage['errorInfo'] = str(e)
            infoOfImage['errorCode'] = e.code
            infoOfImage['flag'] = 1 # 1 is  the image preprocess error
        if infoOfImage.get('flag',0) == 1: # the image preProcess error
            imageReadInfoDict[i] = infoOfImage
            continue
        height, width, _ = img.shape
        infoOfImage['flag'] = 0 # normal image preProcess
        infoOfImage['height'] = height
        infoOfImage['width'] = width
        infoOfImage['normalImageIndex'] = normalImageIndex # because , some images error, so need all images's map relation .
        netinfeImageIdToinputImageId[normalImageIndex] = i # new image id map to old image id
        imageReadInfoDict[i] = infoOfImage
        normalImageIndex += 1
        images.append(preProcessImage(oriImage=img))
    _t2 = time.time()
    CTX.logger.info("read image and transform: %f\n", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)
    return imageReadInfoDict, netinfeImageIdToinputImageId, images


def post_eval(imageIdMapDict,imageReadInfoDict,output, thresholds, label_list):
    '''
        parse net output, as numpy.mdarray, to EvalResponse
        imageIdMapDict : input net image id  --- original input image id 
    '''
    resps = []
    _t1 = time.time()
    # output_bbox_list : bbox_count * 7
    output_bbox_list = output['detection_out'][0][0]
    image_result_dict = dict()  # image_id : bbox_list
    for i_bbox in output_bbox_list:
        # i_bbox : length == 7 ; 0==image_id,1==class_index,2==score,3==bbox_xmin,4==bbox_ymin,5==bbox_xmax,6==bbox_ymax
        image_id = int(i_bbox[0])
        if image_id >= len(imageIdMapDict):
            break
        inputImageId = imageIdMapDict[image_id]
        h = imageReadInfoDict[inputImageId]['height']
        w = imageReadInfoDict[inputImageId]['width']
        class_index = int(i_bbox[1])
        if class_index < 1 : # background index == 0 , refinedet not output background info ,so the line not used
            continue
        score = float(i_bbox[2])
        if score < thresholds[class_index]:
            continue
        name = label_list[class_index]
        bbox_dict = dict()
        bbox_dict['index'] = class_index
        bbox_dict['score'] = score
        bbox_dict['class'] = name
        bbox = i_bbox[3:7] * np.array([w, h, w, h])
        bbox_dict['pts'] = []
        xmin = int(bbox[0]) if int(bbox[0]) > 0 else 0
        ymin = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xmax = int(bbox[2]) if int(bbox[2]) < w else w
        ymax = int(bbox[3]) if int(bbox[3]) < h else h
        bbox_dict['pts'].append([xmin, ymin])
        bbox_dict['pts'].append([xmax, ymin])
        bbox_dict['pts'].append([xmax, ymax])
        bbox_dict['pts'].append([xmin, ymax])
        if image_id in image_result_dict.keys():
            the_image_bbox_list = image_result_dict.get(image_id)
            the_image_bbox_list.append(bbox_dict)
            pass
        else:
            the_image_bbox_list = []
            the_image_bbox_list.append(bbox_dict)
            image_result_dict[image_id] = the_image_bbox_list
    # merge net infe output data and data preProcess result info to resps
    resps = []
    for original_image_id in range(len(imageReadInfoDict)):
        if imageReadInfoDict[original_image_id]['flag'] == 1: # original input image error
            resps.append({
                "code": imageReadInfoDict[original_image_id]['errorCode'], 
                "message": imageReadInfoDict[original_image_id]['errorInfo'],
                "result": {}
            })
            continue
        inputNet_image_id = imageReadInfoDict[original_image_id]['normalImageIndex']
        if inputNet_image_id in image_result_dict.keys():
            res_list = image_result_dict.get(inputNet_image_id)
        else:
            res_list = []
        result = {"detections": res_list}
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f\n", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


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

    CTX.logger.info('detection_out: {}'.format(output['detection_out']))
    if 'detection_out' not in output or len(output['detection_out']) < 1:
        raise ErrorForwardInference()
    return output
