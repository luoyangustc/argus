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
import numpy as np

from evals.utils import create_net_handler, net_inference_handler, CTX, \
                  monitor_rt_load, monitor_rt_forward, monitor_rt_post,\
                  infer_output_marshal,value_check,file_check
from evals.utils.image import load_image

from evals.video_classify.featureCoding import FeatureCoding
from evals.video_classify.config import  config

@create_net_handler
def create_net(configs):
    '''
        net init
    '''
    _t1 = time.time()

    try:
        CTX.logger.debug("enter net_init")
        CTX.logger.debug("configs:%s",configs)
        batch_size = value_check(configs, 'batch_size', False, 1)
        tar_files = value_check(configs,"model_files")
        synset = str(file_check(tar_files,"lsvc_class_index.txt"))
        prefix=os.path.abspath(synset+"/..")+"/netvlad"

        net =  FeatureCoding(featureDim=config.FEATURE_CODING.FEATURE_DIM,
	                               batchsize=batch_size,
	                               modelPrefix=prefix,
									modelEpoch=config.FEATURE_CODING.MODEL_EPOCH,
									synset=synset, gpu_id=0)
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
    datas = []
    rets = []
    features= []
    length=-1
    _t1=time.time()

    datas = reqs[0].get("data",[])    
    if len(datas) == 0:
        rets.append({"code": 400, "message": "no enough params provided"})
        return rets,406,''

    try:
        for data in datas:

            if data['uri'] is None and data['body'] is None:
                CTX.logger.debug("read data failed, None")
                rets.append({"code": 400, "message": "read data failed"})
                return rets,400,''
            
            feature=None
            if data['body'] is not None: 
                feature=struct.unpack('>' + str(len(data['body'])/4) + 'f',data['body'])
            else:
                file=open(data['uri'],'rb')
                fdt=file.read()
                feature=struct.unpack('>' + str(len(fdt)/4) + 'f',fdt)
                file.close()
            if length == -1:
                length = len(feature)
            if feature is None or len(feature) == 0 or len(feature)!=length:
                CTX.logger.debug("json.loads failed:%s",str(len(feature)))
                rets.append({"code": 400, "message": "load data failed"})
                return rets,400,''
            features.append(feature)
        features=np.asarray(features,dtype=np.float32)
        ret = handler.classify(features)
        if ret is None:
            rets.append({"code": 400, "message": "inference failed"}) 
            return rets, 599, ''
        ret=[{x:y} for x, y in sorted(ret.items(),key=lambda item:item[1],reverse=True)]
        rets.append({"code": 0, "message": "success","result":ret})
    except Exception as e:
        CTX.logger.error("inference error:%s",traceback.format_exc())
        return [], 599, str(e)
    _t2=time.time()
    monitor_rt_forward().observe(_t2-_t1)

    CTX.logger.debug("rets:%s",rets)
    return rets, 0, '' 