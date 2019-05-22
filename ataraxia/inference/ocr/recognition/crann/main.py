import pyximport
pyximport.install()
import json
from evals.src.crannRec.crannrec import CrannRecModel
import cv2
import numpy as np
import base64
import config
import time
import traceback
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
        monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    # crann_recog = CrannRecModel(config.MODELFILE_CRANN_URL, config.MODELFILE_CRANN_YML_URL)
    crann_recog = CrannRecModel(str(configs['model_files']['crann_exp3_11_0_3.pth']), str(configs['model_files']['crann.yml']))
    return {"crann_recog": crann_recog, "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    crann_recog = model['crann_recog']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")

    try:
        images_with_bboxes = pre_eval(batch_size, reqs)
        output = eval(crann_recog, images_with_bboxes)
        ret = post_eval(output)
            
    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)

    return ret, 0, ''

def pre_eval(batch_size, reqs):
    cur_batchsize = len(reqs)
    if cur_batchsize > batch_size:
        raise ErrorOutOfBatchSize(batch_size)

    ret = []
    _t1 = time.time()
    for req in reqs:
        img = load_image(req["data"]["uri"], body=req['data']['body'])
        bboxes = req["params"]["bboxes"]
        if len(bboxes[0]) == 8:
            # transform from [x0,y0,x1,y1,x2,y2,x3,y3] to [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            bboxes = list(map(lambda x:[[x[0],x[1]],[x[2],x[3]],[x[4],x[5]],[x[6],x[7]]],bboxes))
        ret.append((img, bboxes))

    _t2 = time.time()
    CTX.logger.info("image load cost: %f", _t2 - _t1)
    # monitor_rt_load().observe(_t2 - _t1)

    return ret


def eval(crann_recog, imges_with_bboxes):
    output = []
    _t1 = time.time()
    for img, rects in imges_with_bboxes:
        imglist = crann_recog.cutimagezz(img,rects)
        res = crann_recog.deploy(imglist)
        output.append((res, rects))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output


def post_eval(output):
    resps = []

    _t1 = time.time()
    for res, bboxes in output:
        result = {}
        result["text"] = res
        result["bboxes"] = bboxes
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


if __name__ == '__main__':
    # init model
    pass