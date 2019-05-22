import pyximport; pyximport.install()
import argparse
from cfg import Config
from timer import Timer
from src.recognizers import TextRecognizer
from src.other import rank_boxes, post_processing
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
    text_recognizer = TextRecognizer(str(mode_path), Config.TEXT_RECOG_ALPHABET)
    return text_recognizer


def text_recog(text_recognizer, text_lines, im):
    timer = Timer()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    timer.tic()
    text_lines = rank_boxes(text_lines)
    predictions = text_recognizer.predict(im, text_lines)
    print "Recognition Time: %f" %timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    return predictions, text_lines

def rank_boxes(boxes):
    def getKey(item):
        return item[1] #sort by y1
    sorted_boxes = sorted(boxes,key=getKey)
    return sorted_boxes

def combine_text(text, text_bboxes, img_type):
    func = post_processing(img_type)
    text_all = func(text, text_bboxes)
    return text_all


def load_text_bboxes(text_path):
    with open(text_path, 'r') as f:
        line = f.read()
        det_result = json.loads(line)
        text_lines = det_result['bboxes']
        img_type = det_result['img_type']
    return text_lines, img_type



def dump_result(text_pred):
    #json_result = json.dumps(text_pred, ensure_ascii=False)
    return text_pred


@create_net_handler
def create_net(configs):

    CTX.logger.info("load configs: %s", configs)
    Config.PLATFORM = configs['use_device'].upper()
    text_recognizer = init_models(configs['model_files']['weight.caffemodel'])
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
        imges_with_bboxes_type = pre_eval(batch_size, reqs)
        output = eval(text_recognizer, imges_with_bboxes_type)
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
    for i in range(cur_batchsize):
        img = load_image(reqs[i]["data"]["uri"], body=reqs[i]['data']['body'])
        bboxes = reqs[i]["params"]["bboxes"]
        img_type = reqs[i]["params"]["image_type"]
        ret.append((img, bboxes, img_type))

    _t2 = time.time()
    CTX.logger.info("load: %f", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    return ret


def post_eval(text_recognizer, output, reqs=None):

    resps = []
    cur_batchsize = len(output)
    _t1 = time.time()
    for i in xrange(cur_batchsize):
        predictions = output[i][0]
        text_bboxes = output[i][1]
        img_type = output[i][2]
        text_recog_result = combine_text(predictions, text_bboxes, img_type)
        result = dump_result(text_recog_result)
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(text_recognizer, imges_with_bboxes_type):
    output = []
    _t1 = time.time()
    for i in range(len(imges_with_bboxes_type)):
        predictions, text_bboxes = text_recog(text_recognizer, imges_with_bboxes_type[i][1], imges_with_bboxes_type[i][0])
        output.append((predictions, text_bboxes, imges_with_bboxes_type[i][2]))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output
