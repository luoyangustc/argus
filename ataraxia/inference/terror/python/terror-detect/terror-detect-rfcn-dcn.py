# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
sys.path.insert(0, os.path.join('/opt/dcn', 'rfcn'))
import rfcn._init_paths

import time
import traceback
import numpy as np
import mxnet as mx
import cv2
import json

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image

from rfcn.symbols import *
from rfcn.config.config import config, update_config
from core.tester import Predictor, im_detect, im_proposal, vis_all_detection, draw_all_detection
from lib.utils.load_model import load_param
from lib.utils.image import resize, transform
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from nms.box_voting import py_box_voting_wrapper

from evals.utils import _make_synset, infer_output_marshal
from evals.utils.image import load_image
from evals.utils.error import ErrorBase
from evals import net

@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    # configs = json.loads(configs)
    tar_files_name='model_files'
	# load tar files 
    if tar_files_name not in configs:
        return None, 400, 'no field tar_files'
    tar_files = configs[tar_files_name]
    conf, err = net.parse_infer_config(tar_files)
    if err:
        return None, 400, err

    params_file, sym_file, label_file = (conf.weight, conf.deploy_sym, conf.labels)
    use_device_name = 'use_device'
    if use_device_name not in configs:
        return None, 400, 'no field use_devices'
    use_device = configs[use_device_name]

    ctx = [mx.gpu()] if use_device == 'GPU' else [mx.cpu()]
    labels = parse_label_file(label_file)

    os.rename(sym_file, sym_file+'-symbol.json')
    os.rename(params_file, sym_file+'-0000.params')

    yaml_file = "/workspace/serving/python/evals/resnet.yaml"

    if os.path.isfile(yaml_file):
        update_config(yaml_file)
    else:
        return None, 400, 'No yaml files'


    predictor=get_net(config, ctx, sym_file, config.TEST.test_epoch, True)
    if predictor==None:
        return None, 400, 'fail to create predictor'

    return dict(
        error='',
        predictor=predictor,
        labels=labels), 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''

@net_inference_handler
def net_inference(model, reqs):

    CTX.logger.info("inference begin...")
    # datas = json.loads(args)
    predictor = model['predictor']
    classes_dict = model['labels']['class']
    # threshold uses for default 
    threshold_dict = model['labels']['minMt'] # minModelThreshold
    rets = []
    nms = py_nms_wrapper(config.TEST.NMS)
    box_voting = py_box_voting_wrapper(config.TEST.BOX_VOTING_IOU_THRESH, config.TEST.BOX_VOTING_SCORE_THRESH,
                                      with_nms=True)

    try:
        for data in reqs:
            try:
                im = load_image(data['data']['uri'], body=data['data']['body'])
            except ErrorBase as e:
                rets.append({"code":e.code, "message": e.message, "result": None})
                continue
                # return [], 400, 'load image error'

            if im.shape[0] > im.shape[1]:
                long_side, short_side = im.shape[0], im.shape[1]
            else:
                long_side, short_side = im.shape[1], im.shape[0]

            if short_side > 0 and float(long_side)/float(short_side) > 50.0:
                msg = "aspect ration is too large, long_size:short_side should not larger than 50.0"
                # raise ErrorBase.__init__(400, msg)
                rets.append({"code": 400, "message": msg, "result": None})
                continue

            data_batch, data_names, im_scale = generate_batch(im)
            scores, boxes, data_dict = im_detect(predictor,
                                                data_batch,
                                                data_names,
                                                im_scale,
                                                config)
            det_ret = []
            # labels.csv file not include background
            for cls_index in sorted(classes_dict.keys()):
                cls_ind = cls_index
                cls_name = classes_dict.get(cls_ind)
                cls_boxes = boxes[0][:, 4:8] if config.CLASS_AGNOSTIC else boxes[0][:, 4 * cls_ind:4 *4 * (cls_ind + 1)]
                cls_scores = scores[0][:, cls_ind, np.newaxis]
                threshold = float(threshold_dict[cls_ind])
                keep = np.where(cls_scores > threshold)[0]
                dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                keep = nms(dets)
                det_ret.extend(_build_result(det, cls_name, cls_ind, model['labels'])
                    for det in dets[keep, :])
            # get review value
            rets.append(dict(code=0,message='',result=dict(detections=det_ret)))

    except Exception as e:
        # print(traceback.format_exc())
        CTX.logger.info("inference error:%s"%(traceback.format_exc()))
        return [], 599, str(e)
    return rets, 0, ''


def get_net(cfg, ctx, prefix, epoch, has_rpn):
    try:
        if has_rpn:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol(cfg, is_train=False)
        else:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)

        # load model
        arg_params, aux_params = load_param(prefix, epoch, process=True)

        # infer shape
        SHORT_SIDE = config.SCALES[0][0]
        LONG_SIDE = config.SCALES[0][1]
        DATA_NAMES = ['data', 'im_info']
        LABEL_NAMES = None
        DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
        LABEL_SHAPES = None
        data_shape_dict = dict(DATA_SHAPES)
        sym_instance.infer_shape(data_shape_dict)
        sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

        # decide maximum shape
        max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
        if not has_rpn:
            max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

        # create predictor
        predictor = Predictor(sym, DATA_NAMES, LABEL_NAMES,
                            context=ctx, max_data_shapes=max_data_shape,
                            provide_data=[DATA_SHAPES], provide_label=[LABEL_SHAPES],
                            arg_params=arg_params, aux_params=aux_params)
    except Exception as e:
        print(traceback.format_exc())
        predictor =  None

    return predictor

def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    SHORT_SIDE = config.SCALES[0][0]
    LONG_SIDE = config.SCALES[0][1]
    PIXEL_MEANS = config.network.PIXEL_MEANS
    DATA_NAMES = ['data', 'im_info']

    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [[mx.nd.array(im_array), mx.nd.array(im_info)]]
    data_shapes = [[('data', im_array.shape), ('im_info', im_info.shape)]]
    data_batch = mx.io.DataBatch(data=data, label=[None], provide_data=data_shapes, provide_label=[None])

    return data_batch, DATA_NAMES, [im_scale]

def _build_result(det, cls_name, cls_ind,labels_csv):
    decodeScoreValue = encodeValue(label_dict=labels_csv, classIndex=int(cls_ind), value=float(det[-1]))
    ret = dict(index=cls_ind, score=decodeScoreValue)
    ret['class'] = cls_name
    x1, y1, x2, y2 = det[:4]
    ret['pts'] = [
	    [int(x1), int(y1)],
	    [int(x2), int(y1)],
	    [int(x2), int(y2)],
	    [int(x1), int(y2)],
	]
    return ret

def parse_label_file(labelFile=None):
    label_dict = dict()
    with open(labelFile, 'r') as f:
        line_list = [i.strip() for i in f.readlines() if i]
        keyList = line_list[0].split(',')  # index,class,threshold
        for key in keyList[1:]:
            label_dict[key] = dict()
        for i_line in line_list[1:]:
            i_line_list = i_line.split(',')
            index_value = int(i_line_list[0])
            for colume_index, value in enumerate(i_line_list[1:], 1):
                label_dict[keyList[colume_index]][index_value] = value
    return label_dict

def encodeValue(label_dict=None,classIndex=None,value=None):
    """
        input : 
                label_dict : dict ,labels.csv file parse result dict
                classIndex : int ,index of class in model
        return encoded value
    """
    classIndex = int(classIndex)
    value = float(value)
    minModelThreshold = float(label_dict['minMt'][classIndex])
    reviewModelThreshold = float(label_dict['revMt'][classIndex])
    minServingThreshold = float(label_dict['minSt'][classIndex])
    reviewServingThreshold = float(label_dict['revSt'][classIndex])
    resultValue = value
    if value <= minModelThreshold:
        # assert value > minModelThreshold
        resultValue = value
    elif value <= reviewModelThreshold:
        # review = True
        resultValue = (value - minModelThreshold) * ((reviewServingThreshold - minServingThreshold)/(reviewModelThreshold - minModelThreshold)) + minServingThreshold
    elif value > reviewModelThreshold:
        # review = False
        resultValue = (value - reviewModelThreshold) * ((1-reviewServingThreshold)/(1-reviewModelThreshold)) + reviewServingThreshold
    return resultValue

def decodeValue(label_dict=None,classIndex=None,value=None):
    """
        input : 
            label_dict : labels.csv file parse result dict
            classIndex : index of class in model
        return decoded value
    """
    classIndex = int(classIndex)
    value = float(value)
    minModelThreshold = float(label_dict['minMt'][classIndex])
    reviewModelThreshold = float(label_dict['revMt'][classIndex])
    minServingThreshold = float(label_dict['minSt'][classIndex])
    reviewServingThreshold = float(label_dict['revSt'][classIndex])
    resultValue = value
    if value < minServingThreshold :
        resultValue = value
    elif value <= reviewServingThreshold:
        resultValue = (value - minServingThreshold) * ((reviewModelThreshold-minModelThreshold)/(reviewServingThreshold-minServingThreshold)) + minModelThreshold
    elif value > reviewServingThreshold:
        resultValue = (value - reviewServingThreshold) *((1-reviewModelThreshold)/(1-reviewServingThreshold)) + reviewModelThreshold
    return resultValue


