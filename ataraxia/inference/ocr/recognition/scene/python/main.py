import sys

sys.path.insert(0, "./src")

import pyximport

pyximport.install()
import argparse
from config import CONFIG as Config
from timer import Timer
from infer import TextRecognizer
import cv2
import json
import sys
import traceback
import time

reload(sys)
sys.setdefaultencoding('utf-8')

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


def init_models(mode_path):
    text_recognizer = TextRecognizer(str(mode_path), Config.alphabet)
    return text_recognizer


def text_recog(text_recognizer, img, rects):
    timer = Timer()
    print("----------------------------------")
    timer.tic()
    predictions = text_recognizer.infer(img, rects)
    print("Recognition Time: %f" % timer.toc())
    print("----------------------------------")

    return predictions


@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    Config.PLATFORM = configs['use_device'].upper()
    text_recognizer = init_models(configs['model_files']['crnn_0_8.pth'])
    return {"text_recognizer": text_recognizer, "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    text_recognizer = model['text_recognizer']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")

    try:
        images_with_pts = pre_eval(batch_size, reqs)
        output = eval(text_recognizer, images_with_pts)
        ret = post_eval(text_recognizer, output, reqs)

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
        detections = req["data"]["attribute"]["detections"]
        pts_arr = []
        for det in detections:
            pts_arr.append(det["pts"])

        ret.append((img, pts_arr))

    _t2 = time.time()
    CTX.logger.info("load: %f", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    return ret


def post_eval(text_recognizer, output, reqs=None):
    resps = []

    _t1 = time.time()
    for predictions, text_pts in output:
        result = {}
        items = []

        for prediction, text_bbox in zip(predictions, text_pts):
            item = {}
            item['pts'] = text_bbox
            item['text'] = ' '.join(prediction.split())
            items.append(item)

        result["texts"] = items
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(text_recognizer, imges_with_pts):
    output = []
    _t1 = time.time()
    for img, rects in imges_with_pts:
        predictions = text_recog(text_recognizer, img, rects)
        output.append((predictions, rects))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output
