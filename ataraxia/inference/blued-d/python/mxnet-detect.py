from __future__ import print_function

import json
import os

import cv2
import traceback
import mxnet as mx
import numpy as np
from rcnn.config import config
from rcnn.core.tester import (Predictor, im_detect)
from rcnn.io.image import resize, transform
from rcnn.processing.nms import py_nms_wrapper
from rcnn.symbol import get_resnet_test
from rcnn.utils.load_model import load_param

from evals.utils import *
from evals.utils.error import *
from evals.mxnet_base import net
from evals.utils.image import load_image

config.TEST.HAS_RPN = True
# SHORT_SIDE = config.SCALES[0][0]
# LONG_SIDE = config.SCALES[0][1]
# PIXEL_MEANS = config.PIXEL_MEANS
SHORT_SIDE = 800
LONG_SIDE = 1500
PIXEL_MEANS = np.array([0,0,0])
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
LABEL_SHAPES = None
# visualization
CONF_THRESH = 0.85
NMS_THRESH = 0.3
nms = py_nms_wrapper(NMS_THRESH)


def get_net(symbol, prefix, epoch, ctx):
    arg_params, aux_params = load_param(
        prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(DATA_SHAPES)
    arg_names, aux_names = symbol.list_arguments(
    ), symbol.list_auxiliary_states()
    arg_shape, _, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    aux_shape_dict = dict(zip(aux_names, aux_shape))

    # check shapes
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + \
            str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + \
            str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    predictor = Predictor(
        symbol,
        DATA_NAMES,
        LABEL_NAMES,
        context=ctx,
        provide_data=DATA_SHAPES,
        provide_label=LABEL_SHAPES,
        arg_params=arg_params,
        aux_params=aux_params)
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
    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array(
        [[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [mx.nd.array(im_array), mx.nd.array(im_info)]
    data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(
        data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch, DATA_NAMES, im_scale


def _load_cls(label_file):
    return tuple(e[-1] for e in net.load_labels(label_file))


@create_net_handler
def create_net(configs):

    logger.info("[Python create_net] load configs: %s", configs, extra={"reqid": ""})
    tar_files_name = 'model_files'
    # load tar files
    if tar_files_name not in configs:
        return None,400,{"code": 400, "message": 'no field "tar_files"'}

    tar_files = configs[tar_files_name]
    conf, err = net.parse_infer_config(tar_files)
    if err:
        return None,400,{"code": 400, "message": err}

    params_file, sym_file, label_file = (conf.weight, conf.deploy_sym,
                                         conf.labels)

    use_device_name = 'use_device'
    if use_device_name not in configs:
        return None,400,{"code": 400, "message": 'no field "use_device"'} 
    use_device = configs[use_device_name]

    threshold = CONF_THRESH
    if 'custom_params' in configs:
        custom_values = configs['custom_params']
        if 'threshold' in custom_values:
            threshold = custom_values["threshold"]

    ctx = mx.gpu() if use_device == 'GPU' else mx.cpu()  # TODO set the gpu/cpu
    classes = _load_cls(label_file)
    symbol = get_resnet_test(num_classes=len(classes),
                             num_anchors=config.NUM_ANCHORS)


    os.rename(sym_file, sym_file+'-symbol.json')
    os.rename(params_file, sym_file+'-0000.params')

    logger.info("params_file: %s, sym_file:%s,label_file:%s", params_file, sym_file, label_file, extra={"reqid": ""})
    logger.info("use_device: %s, threshold:%s,classes:%s,symbol:%s", use_device, threshold, classes,symbol, extra={"reqid": ""})

    return dict(
        error='',
        predictor=get_net(symbol, sym_file, 0, ctx),
        classes=classes,
        threshold=threshold),0,None


def _build_result(det, cls_name, cls_ind, im_info):
    """
    Construct result dictionary 
    :param det: detection result
    :param cls_name: class name
    :param cls_ind: class index
    :param im_info: extra image info, (height,width,channel)
    :return: dict of bounding box information
    """
    ret = dict(index=cls_ind, score=float(det[-1].round(4)))
    ret['class'] = cls_name
    x1, y1, x2, y2 = det[:4]
    ret['pts'] = [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]
    ret['area_ratio'] = float((x2 - x1) * (y2 - y1)) / (im_info[0] * im_info[1])

    return ret


def _load_image(req):
    try:
        img = load_image(req["uri"], body=req['body'])
    except Exception as _e:
        logger.info("load image error:%s, trackback:%s",str(_e), traceback.format_exc(), extra={"reqid": ""})
        return None, None, {"code": 400, "message": str(_e)}
    return img, img.shape, None

@net_inference_handler
def net_inference(model, args):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :return: None
    """

    predictor = model['predictor']
    classes = model['classes']
    threshold = model['threshold']
    rets = []
    try:
        for data in args:
            im, im_info, err = _load_image(data['data'])
            if err is not None:
                rets.append(err)
                continue

            data_batch, data_names, im_scale = generate_batch(im)
            scores, boxes, data_dict = im_detect(predictor,
                                                data_batch,
                                                data_names,
                                                im_scale)

            det_ret = []
            for cls_ind, cls_name in enumerate(classes[1:], start=1):
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind, np.newaxis]
                keep = np.where(cls_scores >= threshold)[0]
                dets = np.hstack((cls_boxes,
                                cls_scores)).astype(np.float32)[keep, :]
                keep = nms(dets)
                det_ret.extend(_build_result(det, cls_name, cls_ind, im_info)
                            for det in dets[keep, :])

            rets.append(
                dict(
                    code=0,
                    message='',
                    result=dict(detections=det_ret)))
    except Exception as _e:
        logger.info("inference error:%s", traceback.format_exc(), extra={"reqid": ""})
        return [],599,{"code": 599, "message": str(_e)}
    return rets,0,None