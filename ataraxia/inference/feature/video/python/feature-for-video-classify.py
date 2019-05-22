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
import struct

from evals.utils import create_net_handler, net_inference_handler, CTX, \
                  monitor_rt_load, monitor_rt_forward, monitor_rt_post,\
                  infer_output_marshal,value_check,file_check
from evals.utils.image import load_image

from evals.video_feature.feature_extract import FeatureExtraction
from evals.video_feature.config import  config

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
        prototxt = str(file_check(tar_files,"deploy.prototxt"))  
        model = str(file_check(tar_files,"weight.caffemodel")) 
        net = FeatureExtraction(modelPrototxt=prototxt,
										modelFile=model,
										featureLayer=config.FEATURE_EXTRACTION.FEATURE_LAYER)
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
            im = load_image(data["data"]["uri"],data['data']['body'])
            if im is None:
                CTX.logger.debug("read image failed, path:%s",data['data']['uri'])
                rets.append({"code": 400, "message": "read image failed"})
                continue

            feature = handler.ext_process([im])
            if feature is None or len(feature)==0:
                rets.append({"code": 400, "message": "failed to get features of the image"}) 
                continue
            CTX.logger.info("feature length info:" + str(len(feature))+"len(feature[0]): "+str(len(feature[0])))
            stream = struct.pack('>'+str(len(feature[0]))+'f', *feature[0])
            #CTX.logger.error("struct.unpack info:" + ">" +str(len(stream) / 4) + "f")
            #hash_sha1 = hashlib.sha1()
            #hash_sha1.update(stream)
            #feature_file_name = os.path.join("/tmp/eval/", hash_sha1.hexdigest())
            #file = open(feature_file_name, "wb")
            #file.write(stream)
            #file.close()
            #rets.append({"code": 0, "message": "", "result_file": str(feature_file_name)})
            rets.append({"code": 0, "message": "", "body": stream})
    except Exception as e:
        CTX.logger.error("inference error:%s",traceback.format_exc())
        return [], 599, str(e)
    _t2=time.time()
    monitor_rt_forward().observe(_t2-_t1)

    CTX.logger.debug("rets:%s",rets)
    return rets, 0, '' 