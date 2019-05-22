import pyximport; pyximport.install()
import argparse
import cv2, caffe
from src.cfg import Config
from src.other import draw_boxes, resize_im, CaffeModel
from src.detectors import TextProposalDetector, TextDetector
from src.utils.timer import Timer
import json
import time
import traceback
import os
from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
        monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image
import numpy as np

def rank_boxes(boxes):
    def getKey(item):
        return item[1] #sort by y1
    sorted_boxes = sorted(boxes,key=getKey)
    return sorted_boxes

def init_models(use_divice, net_def_file, model_file):
    if use_divice == "GPU":
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # initialize the detectors
    text_proposals_detector = TextProposalDetector(CaffeModel(net_def_file, model_file))
    text_detector = TextDetector(text_proposals_detector)

    return text_detector


def text_detect(text_detector, im):

    im_small, f,im_height,im_width = resize_im(im, Config.SCALE, Config.MAX_SCALE)

    timer = Timer()
    timer.tic()
    text_lines = text_detector.detect(im_small)
    text_lines = draw_boxes(im_small, text_lines,f,im_height,im_width)
    print "Number of the detected text lines: %s" % len(text_lines)
    print "Detection Time: %f" % timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    return text_lines


def dump_result(text_lines):
    text_detect_result = dict()
    text_detect_result['bboxes'] = text_lines
    return text_detect_result


@create_net_handler
def create_net(configs):

    CTX.logger.info("load configs: %s", configs)
    text_detector = init_models(configs['use_device'].upper(),
                                configs['model_files']['deploy.prototxt'], 
                                configs['model_files']['weight.caffemodel'])
    return {"text_detector": text_detector, "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    text_detector = model['text_detector']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")

    try:
        imges_with_type = pre_eval(batch_size, reqs)
        output = eval(text_detector, imges_with_type)
        ret = post_eval(output, reqs)
            
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
    for i in range(cur_batchsize):
        img = load_image(reqs[i]["data"]["uri"], body=reqs[i]['data']['body'])
        ret.append((img))

    _t2 = time.time()
    CTX.logger.info("load: %f", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    return ret


def post_eval(output, reqs=None):

    resps = []
    cur_batchsize = len(output)
    _t1 = time.time()
    for i in xrange(cur_batchsize):
        text_bboxes = output[i]
        if len(text_bboxes) == 0:
            CTX.logger.info("no text detected")
            resps.append({"code": 0, "message": "", "result": {}})
            continue
        text_bboxes = rank_boxes(text_bboxes)
        array_result = np.array(text_bboxes).reshape(-1, 4, 2)
        result = dump_result(array_result.tolist())

        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(text_detector, imges_no_type):
    output = []
    _t1 = time.time()
    for i in range(len(imges_no_type)):
        text_bboxes = text_detect(text_detector, imges_no_type[i])
        output.append((text_bboxes))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output


if __name__ == '__main__':
    configs = {
        "app": "ctpnapp",
        "use_device": "GPU",
        "batch_size":256
    }
    result_dict,_,_=create_net(configs)
    img_path = "/workspace/imagenet-data/ctpn_ataraxia/test_pic"
    img_list = os.listdir(img_path)
    reqs=[]
    temp_i = 0
    for img_name in img_list:
        reqs_temp = dict()
        reqs_temp["data"]=dict()
        reqs_temp["data"]["uri"]=img_path + img_name
        reqs_temp["data"]["body"]=None
        reqs.append(reqs_temp)
    ret = net_inference(result_dict, reqs)
    print(ret)