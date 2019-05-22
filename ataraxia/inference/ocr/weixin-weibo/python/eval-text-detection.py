import pyximport; pyximport.install()
import argparse
import cv2, caffe
from src.cfg import Config
from src.other import draw_boxes, resize_im, refine_boxes, calc_area_ratio, CaffeModel
from src.detectors import TextProposalDetector, TextDetector
from src.utils.timer import Timer
import json
import time
import traceback

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
        monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image

def init_models(use_divice, net_def_file, model_file):
    if use_divice == "GPU":
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # initialize the detectors
    text_proposals_detector = TextProposalDetector(CaffeModel(net_def_file, model_file))
    text_detector = TextDetector(text_proposals_detector)

    return text_detector


def text_detect(text_detector, im, img_type):
    if img_type == "others":
        return [], 0

    im_small, f = resize_im(im, Config.SCALE, Config.MAX_SCALE)

    timer = Timer()
    timer.tic()
    text_lines = text_detector.detect(im_small)
    text_lines = text_lines / f  # project back to size of original image
    text_lines = refine_boxes(im, text_lines, expand_pixel_len = Config.DILATE_PIXEL,
                              pixel_blank = Config.BREATH_PIXEL, binary_thresh=Config.BINARY_THRESH)
    text_area_ratio = calc_area_ratio(text_lines, im.shape)
    print "Number of the detected text lines: %s" % len(text_lines)
    print "Detection Time: %f" % timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if Config.DEBUG_SAVE_BOX_IMG:
        im_with_text_lines = draw_boxes(im, text_lines, is_display=False, caption=image_path, wait=False)
        if im_with_text_lines is not None:
            cv2.imwrite(image_path+'_boxes.jpg', im_with_text_lines)

    return text_lines, text_area_ratio


def dump_result(text_lines, text_area_ratio, img_type):
    text_detect_result = dict()
    text_detect_result['bboxes'] = text_lines
    text_detect_result['area_ratio'] = text_area_ratio
    text_detect_result['img_type'] = img_type

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
        ret = post_eval(text_detector, output, reqs)
            
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
        img_type = reqs[i]["params"]["image_type"]
        ret.append((img, img_type))

    _t2 = time.time()
    CTX.logger.info("load: %f", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    return ret


def post_eval(text_detector, output, reqs=None):

    resps = []
    cur_batchsize = len(output)
    _t1 = time.time()
    for i in xrange(cur_batchsize):
        text_bboxes = output[i][0]
        text_area_ratio = output[i][1]
        img_type = output[i][2]
        res_list = []
        if len(text_bboxes) == 0:
            CTX.logger.info("no text detected")
            resps.append({"code": 0, "message": "", "result": {}})
            continue
        result = dump_result(text_bboxes, text_area_ratio, img_type)
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(text_detector, imges_with_type):
    output = []
    _t1 = time.time()
    for i in range(len(imges_with_type)):
        text_bboxes, text_area_ratio = text_detect(text_detector, imges_with_type[i][0], imges_with_type[i][1])
        output.append((text_bboxes, text_area_ratio, imges_with_type[i][1]))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output
