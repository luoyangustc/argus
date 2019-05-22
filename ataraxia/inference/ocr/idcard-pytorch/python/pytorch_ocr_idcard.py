# -*- coding: utf-8 -*-

import base64
import sys
import io
import json
import os
import cv2
import time
import traceback
import hashlib

from evals.utils import create_net_handler, net_inference_handler, CTX, \
                  monitor_rt_load, monitor_rt_forward, monitor_rt_post,\
                  infer_output_marshal,value_check,file_check
from evals.utils.image import load_image

from pytorch_idcard.id_card import IDCardRecognizeHandler
from pytorch_idcard import id_card

@create_net_handler
def create_net(configs):
    '''
        net init
    '''
    _t1 = time.time()

    try:
        CTX.logger.debug("enter net_init")
        CTX.logger.debug("configs:%s",configs)
        tar_files = value_check(configs,"model_files")
        digit_model = file_check(tar_files,"digits_model.pth")  
        crnn_model = file_check(tar_files,"netcrnn_model.pth") 
        id_card.config.RECOGNITION.ADDRESS_MODEL_PATH =  crnn_model
        id_card.config.RECOGNITION.NAME_MODEL_PATH = crnn_model
        id_card.config.RECOGNITION.DIGITS_MODEL_PATH = digit_model
        id_card.config.PLATFORM=value_check(configs, 'use_device',default="GPU").upper()
        net = IDCardRecognizeHandler()
    except Exception as e:
        CTX.logger.error("load error:%s",traceback.format_exc())
        return {}, 599, str(e)
    _t2 = time.time()
    monitor_rt_load().observe(_t2-_t1)
    CTX.logger.info("load time:%f",_t2-_t1)

    return {"net":net}, 0, ''

@net_inference_handler
def net_inference(model, reqs):
    '''
        net inference
    '''
    CTX.logger.debug("enter net_inference")
    handler = model['net']
    rets = []

    _t1=time.time()

    try:
        for data in reqs:
            CTX.logger.debug("data:%s",data)
            img=data["data"]["uri"]
            if data['data']['body'] is not None:
                hash_sha1 = hashlib.sha1()
                hash_sha1.update(str(data['data']['body']))
                img = os.path.join("/tmp", hash_sha1.hexdigest())
                file = open(img, "wb")
                file.write(data['data']['body'])
                file.close()
            im = load_image(img)
            if im is None:
                CTX.logger.debug("read image failed, path:%s",data['data']['uri'])
                rets.append({"code": 400, "message": "read image failed"})
                continue

            ret = handler.recognize(im)
            if ret["status"]==-1:
                rets.append({"code": 400, "message": "no valid id info obtained"}) 
                continue
            rets.append(dict(
                code=0,
                message='',
                result=ret['id_res']
            ))
            if data['data']['body'] is not None and os.path.exists(img):
                os.remove(img)
    except Exception as e:
        if data['data']['body'] is not None and os.path.exists(img):
            os.remove(img)
        CTX.logger.error("inference error:%s",traceback.format_exc())
        return [], 599, str(e)
    _t2=time.time()
    monitor_rt_forward().observe(_t2-_t1)

    CTX.logger.debug("rets:%s",rets)
    return rets, 0, '' 
