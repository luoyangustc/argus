#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-08-10 14:37
from collections import namedtuple
import json
import os
import random
import cv2
import mxnet as mx
import numpy as np

from evals import utils
from evals.utils import *
from evals.utils.error import *
from evals.utils.image import load_image
from evals.utils.logger import logger
from evals.mxnet_base import net
#import time
from evals.utils import monitor_rt_transform, monitor_rt_load, monitor_rt_forward, monitor_rt_post

_mean_value=[123.68,116.779,103.939]
_mean_std=[58.395,57.12,57.375]
_batch_size=5
_map_pre_score=0.89

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

        ctx = mx.gpu()# if conf.use_device == 'GPU' else mx.cpu()

        sym, arg_params, aux_params = mx.model.load_checkpoint(sym_file, 0)
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

        logger.info("config of width:{}, height:{}, value_mean:{}, value_std:{},batch_size:{}".format(conf.image_width,conf.image_height,conf.value_mean,conf.value_std,conf.batch_size), extra={"reqid": ""})
        if conf.image_width==0 or conf.image_width==None:
            conf.image_width=conf.image_height
        if conf.image_height==0 or conf.image_height==None:
            conf.image_height=conf.image_width
        if conf.value_mean == None:
            conf.value_mean = _mean_value
        if conf.value_std == None:
            conf.value_std =  _mean_std
        if conf.batch_size == None or conf.batch_size == 0:
            conf.batch_size = _batch_size

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
            if (type(data["params"]["limit"]) is int or data["params"]["limit"].isdigit()) and int(
                    data["params"]["limit"]) <= len(labels):
                limit = int(data["params"]["limit"])
            else:
                rets[i] = {"code": 400, "message": "invalid limit params"}
                continue
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
        return rets, 0, None
    except Exception as _e:
        logger.error("_load_image error: %s", traceback.format_exc(), extra={"reqid": ""})
        return None, 400, {"code": 400, "message": str(_e)}


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

def _score_map(score):
    map_pre_score = _map_pre_score
    map_post_score = 0.6
    if score > map_pre_score:
        a = (1 - map_post_score) / (1 - map_pre_score)
        b = 1 - a
    else:
        a = map_post_score / map_pre_score
        b = 0
    score = (a * score + b)
    return score

def _build_result(output, labels, limit=1):
    results = {"confidences": [],
                "checkpoint":None}

    label_dic = {0:2,2:1,1:0}
    normal_score = float(output[0])
    if normal_score >=0.960:
        a = random.uniform(0,1)
        b = 1.0-a
        mock = 1.0-normal_score
        pulp_score = min(a,b)*mock
        sexy_score = max(a,b)*mock
        results["checkpoint"]="endpoint"

        results["confidences"].append(
                {
                    'score':_score_map(float(output[0])),
                    'class':labels[0][-1],
                    'index':int(2)
                })
        results["confidences"].append(
                {
                    'score':_score_map(sexy_score),
                    'class':labels[2][-1],
                    'index':int(1)
                })
        results["confidences"].append(
                {
                    'score':_score_map(pulp_score),
                    'class':labels[1][-1],
                    'index':int(0)
                })

    else:
        results["checkpoint"]="ava-pulp"
        j = np.argsort(output)[::-1]

        results["confidences"].append(
                {
                    'score':_score_map(float(output[j[0]])),
                    'class':labels[j[0]][-1],
                    'index':label_dic[int(labels[j[0]][0])]
                })
        results["confidences"].append(
                {
                    'score':_score_map(float(output[j[1]])),
                    'class':labels[j[1]][-1],
                    'index':label_dic[int(labels[j[1]][0])]
                })
        results["confidences"].append(
                {
                    'score':float(0),
                    'class':'sexy',
                    'index':1
                })

    return results

def unit_test():
    configs = {
        "tar_files":{
            'deploy.symbol.json': '/workspace/serving/cache/model_test/deploy.symbol.json',
            'weight.params': '/workspace/serving/cache/model_test/weight.params',
            'labels.csv': '/workspace/serving/cache/model_test/labels.csv'
        },
        "use_device": 'gpu',
        "batch_size": 5,
        "image_width": 224,
        "custom_values":{
            "image_height": 224,
            "mean_value": [123.68,116.779,103.939],
            "mean_std": [58.395,57.12,57.375]
        }
    }
    # imgs = [
    #     {
    #         "data":
    #         {
    #             "uri": "/workspace/serving/run/cache/test_0.jpg",
    #             "body": None
    #         }
    #     },
    #     {
    #         "data":
    #         {
    #             "uri": "/workspace/serving/run/cache/test_1.jpg",
    #             "body": None
    #         }
    #     }
    # ]
    imgs = []
    for img in os.listdir('/workspace/serving/cache/pulp_set1/'):
        with open('/workspace/serving/cache/pulp_set1/' + img, 'r') as f:
            body = f.read()
        imgs.append({
            "data":
            {
                "uri": "/workspace/serving/cache/pulp_set1/" + img,
                "body": body
            },
            "params":
            {
                "limit":3
            }
        })
    nets, _, _ = create_net(configs)
    cnt = 0
    batch_size = 5
    while True:
        start = cnt * batch_size
        end = (cnt + 1) * batch_size
        cnt += 1
        if end > len(imgs):
            end = len(imgs)
        rets, _, _ = net_inference(nets, imgs[start: end])
        with open('reg.tsv', 'a') as f:
            for i, ret in enumerate(rets):
                ret['result']['confidences'].sort(key=lambda x : x['score'])
                f.write(imgs[i + start]['data']['uri'].split('/')[-1] + '\t')
                f.write(json.dumps([ret['result']['confidences'][-1]]))
                f.write('\n')
        if end == len(imgs):
            break

if __name__ == '__main__':
    unit_test()