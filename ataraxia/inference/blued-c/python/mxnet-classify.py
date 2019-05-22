#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-08-10 14:37
from collections import namedtuple
import json
import os

import cv2
import mxnet as mx
import numpy as np

from evals import utils
from evals.utils import *
from evals.utils.error import *
from evals.utils.image import load_image
from evals.utils.logger import logger
from evals.mxnet_base import net

from evals.utils import monitor_rt_transform, monitor_rt_load, monitor_rt_forward, monitor_rt_post

@create_net_handler
def create_net(configs):
    '''
    the params file format: module_name + others + '-dddd.params'
    'dddd' represents 4 digits
    '''
    conf = net.NetConfig()
    conf.parse(configs)

    logger.info("[Python net_init] load configs: %s", configs, extra={"reqid": ""})

    try:
        params_file, sym_file, label_file = (conf.file_model, conf.file_symbol,
                                             conf.file_synset)
        os.rename(sym_file, sym_file + '-symbol.json')
        os.rename(params_file, sym_file + '-0000.params')

        ctx = mx.gpu() if conf.use_device == 'GPU' else mx.cpu()

        sym, arg_params, aux_params = mx.model.load_checkpoint(sym_file, 0)
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

        logger.info("config of width:{}, height:{}, value_mean:{}, value_std:{},batch_size:{}".format(conf.image_width,conf.image_height,conf.value_mean,conf.value_std,conf.batch_size), extra={"reqid": ""})
        
        default_width=224
        if conf.image_width==0 or conf.image_width==None:
            conf.image_width=default_width
        if conf.image_height==0 or conf.image_height==None:
            conf.image_height=conf.image_width
            
        mod.bind(for_training=False,
                 data_shapes=[('data', (conf.batch_size, 3, conf.image_width, conf.image_height))],
                 label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True)

    except Exception as _e:
        logger.info("[Python net_init] failed: %s", traceback.format_exc(), extra={"reqid": ""})
        return {}, 599, str(_e)

    return {"net": dict(
        error='',
        labels=net.load_labels(label_file),
        image_width=conf.image_width,
        image_height=conf.image_height,
        mean_value=conf.value_mean,
        std_value=conf.value_std,
        batch_size=conf.batch_size,
        mod=mod)}, 0,  ''


@net_inference_handler
def net_inference(model, args):
    model = model["net"]
    mod, labels = model['mod'], model['labels']
    width, height = model['image_width'], model['image_height']
    batch_size = model['batch_size']

    mean_value = [0.0, 0.0, 0.0]
    std_value = [1.0, 1.0, 1.0]
    if type(model["mean_value"]) is list and len(model["mean_value"]) == 3:
        mean_value = model["mean_value"]
    if type(model["std_value"]) is list and len(model["std_value"]) == 3:
        std_value = model["std_value"]

    if len(args) > batch_size:
        for i in range(cur_batchsize):
            raise ErrorOutOfBatchSize(batch_size)

    Batch = namedtuple('Batch', ['data'])
    rets = range(len(args))
    valid_imgs = []

    for i, data in enumerate(args):
        im, err = _load_image(data['data'], width, height, mean_value, std_value)
        if err:
            logger.error("_load_image error : %s", data['data']['uri'], extra={"reqid": ""})
            rets[i] = err
            continue
        limit = 1
        if "params" in data and "limit" in data["params"]:
            if (type(data["params"]["limit"]) is int or data["params"]["limit"].isdigit()) and \
                int(data["params"]["limit"]) <= len(labels) and int(data["params"]["limit"]) > 0:
                limit = int(data["params"]["limit"])

        valid_imgs.append(dict(index=i, img=im, limits=limit))
    if len(valid_imgs) == 0:
        return rets, 400, None

    try:
        img_batch = mx.nd.array(np.zeros((batch_size, 3, width, height)))
        for index, image in enumerate(valid_imgs):
            img_batch[index] = mx.nd.array(image["img"])

        mod.forward(Batch([img_batch]))
        output_batch = mod.get_outputs()[0].asnumpy()
        ## post process
        for i in xrange(len(valid_imgs)):
            rets[valid_imgs[i]["index"]] = dict(
                code=0,
                message='',
                result=_build_result(output_batch[i], labels, valid_imgs[i]["limits"])
            )
            
    except Exception as _e:
        logger.error("_load_image error: %s", traceback.format_exc(), extra={"reqid": ""})
        return None, 400, {"code": 400, "message": str(_e)}
    return rets, 0, None


def _load_image(req, width, height, mean_value=[0.0, 0.0, 0.0], std_value=[1.0, 1.0, 1.0]):
    try:
        img = load_image(req["uri"], body=req['body'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            return None, {"code": 400, "message": "cv2 load image failed"}

            # convert into format (batch, RGB, width, height)
        img = img.astype(float)
        img = cv2.resize(img, (width, height))
        img -= mean_value
        img /= std_value
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        return img, None
    except Exception as _e:
        logger.error("_load_image error: %s", traceback.format_exc(), extra={"reqid": ""})
        if isinstance(_e, ErrorBase):
            return None, {"code": _e.code, "message": str(_e)}
        return None, {"code": 400, "message": str(_e)}


def _build_result(output, labels, limit=1):
    results = {"confidences": []}
    for i in xrange(limit):
        j = np.argsort(output)[::-1][i]
        results["confidences"].append(
            {
                'score': float(output[j]),
                'class': labels[j][-1],
                'index': int(labels[j][0])
            }
        )
    return results
