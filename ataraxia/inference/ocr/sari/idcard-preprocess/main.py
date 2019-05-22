# coding:utf-8
import json
# import requests
import config
import time
import base64
import logging
from evals.src.idcard_class import IDCARDCLASS
import config
import cv2
import sys
import numpy as np
from evals.src.IDCardfront.Aligner import AlignerIDCard
from evals.src.IDCardback.Aligner import AlignerIDCardBack

reload(sys)
sys.setdefaultencoding('utf-8')

import traceback
from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    idcardcls = IDCARDCLASS(config.ALIGNER_TEMPLATE_IDCARDCLASS_IMG_PATH,
                            config.ALIGNER_TEMPLATE_IDCARDCLASSBACK_IMG_PATH)
    idcardfront = AlignerIDCard(
        config.ALIGNER_TEMPLATE_IDCARD_IMG_PATH, config.ALIGNER_TEMPLATE_IDCARD_LABEL_PATH)
    idcardback = AlignerIDCardBack(
        config.ALIGNER_TEMPLATE_IDCARDBACK_IMG_PATH, config.ALIGNER_TEMPLATE_IDCARDBACK_LABEL_PATH)

    return {"idcardcls": idcardcls, "idcardfront": idcardfront, "idcardback": idcardback, "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    idcardcls = model['idcardcls']
    idcardfront = model['idcardfront']
    idcardback = model['idcardback']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")

    ret = []
    cur_batchsize = len(reqs)
    if cur_batchsize > batch_size:
        raise ErrorOutOfBatchSize(batch_size)

    try:
        for req in reqs:
            if req["params"]["type"] == "predetect":
                result = predetect(idcardcls, idcardfront,
                                   idcardback, batch_size, req)
            elif req["params"]["type"] == "prerecog":
                if req["params"]["class"] == 0:
                    result = prerecog(idcardfront, batch_size, req)
                else:
                    result = prerecog(idcardback, batch_size, req)
            elif req["params"]["type"] == "postprocess":
                if req["params"]["class"] == 0:
                    result = postprocess(idcardfront, batch_size, req)
                else:
                    result = postprocess(idcardback, batch_size, req)
            else:
                return [], 400, 'bad request - with wrong params'

        ret.append({"code": 0, "message": "", "result": result})

    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)

    return ret, 0, ''


def predetect(idcardcls, idcardfront, idcardback, batch_size, req):
    _t1 = time.time()
    # loading image
    img = load_image(req["data"]["uri"], body=req['data']['body'])
    _t2 = time.time()
    CTX.logger.info("inference :: cost for loading image: %f", _t2 - _t1)

    # classify idcard
    cls = idcardcls.run(img)
    _t3 = time.time()
    CTX.logger.info("inference :: cost for loading image: %f", _t3 - _t2)

    # preprocess for adjust idcard
    # 0: front side;    1: back side;
    if(cls == 0):
        alignedImg, names, regions, boxes = idcardfront.predet(img)
    else:
        alignedImg, names, regions, boxes = idcardback.predet(img)

    _t4 = time.time()
    CTX.logger.info("inference :: cost for image preprocessing: %f", _t4 - _t3)

    return {
        "class": cls,
        "alignedImg": base64.b64encode(cv2.imencode('.jpg', alignedImg)[1]),
        "names": names,
        "regions": regions,
        "bboxes": boxes
    }


def prerecog(handler, batch_size, req):
    _t1 = time.time()
    # loading image
    img = load_image(req["data"]["uri"], body=req['data']['body'])
    _t2 = time.time()
    CTX.logger.info("inference :: cost for loading image: %f", _t2 - _t1)

    CTX.logger.info("input names: %s", req["params"]["names"])
    names = map(lambda name: name.encode('utf8'), req["params"]["names"])

    detboxes = req["params"]["detectedBoxes"]
    if len(detboxes[0]) == 8:
        # transform from [x0,y0,x1,y1,x2,y2,x3,y3] to [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        detboxes = list(map(lambda x: [[x[0], x[1]], [x[2], x[3]], [
                        x[4], x[5]], [x[6], x[7]]], detboxes))

    bboxes = handler.prerecog(
        detboxes, img, names, req["params"]["regions"], req["params"]["bboxes"])
    _t3 = time.time()
    CTX.logger.info(
        "inference :: cost for image pre-recognize process: %f", _t3 - _t2)

    return {
        "bboxes": bboxes
    }


def postprocess(handler, batch_size, req):
    _t1 = time.time()
    CTX.logger.info("input texts: %s", req["params"]["texts"])
    CTX.logger.info("input names: %s", req["params"]["names"])
    CTX.logger.info("input bboxes: %s", req["params"]["bboxes"])
    texts = map(lambda text: text.encode('utf8'), req["params"]["texts"])
    names = map(lambda name: name.encode('utf8'), req["params"]["names"])
    res = handler.postprocess(
        req["params"]["bboxes"], texts, req["params"]["regions"], names)
    _t2 = time.time()
    CTX.logger.info(
        "inference :: cost for construct id card information: %f", _t2 - _t1)

    CTX.logger.info("origin result texts: %s", res)

    return {
        "res": res
    }


if __name__ == '__main__':
    # init model
    pass
