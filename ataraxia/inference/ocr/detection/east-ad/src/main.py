import pyximport
pyximport.install()
import argparse
import cv2
import base64
import json
import time
import traceback
import numpy as np
import tensorflow as tf
import evals.src.lanms as lanms
from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image
from evals.src.cfg import Config
import os
import evals.src.model as model
from evals.src.icdar import restore_rectangle
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import tensorrt as trt
from tensorrt.parsers import uffparser

def regular_pts(pt1, pt2, pt3, pt4, max_width):
    if pt1[1] != pt2[1] and abs(pt1[1] - pt2[1]) <= 3:
        pt1[1] = pt2[1] = min(pt1[1], pt2[1]) + 1

    if pt3[1] != pt4[1] and abs(pt3[1] - pt4[1]) <= 3:
        pt3[1] = pt4[1] = max(pt3[1], pt4[1]) - 1

    if pt1[0] != pt4[0] and abs(pt1[0] - pt4[0]) <= 3:
        pt1[0] = pt4[0] = min(pt1[0], pt4[0])

    if pt2[0] != pt3[0] and abs(pt2[0] - pt3[0]) <= 3:
        pt2[0] = pt3[0] = max(pt2[0], pt3[0])
    if pt1[1] == pt2[1] and pt3[1] == pt4[1] and 0 < abs(pt4[1] - pt1[1]) < abs(pt3[0] - pt4[0]):
        pt1[0] = max(0, pt1[0] - 6)
        pt4[0] = max(0, pt4[0] - 6)
        pt2[0] = min(max_width - 1, pt2[0] + 6)
        pt3[0] = min(max_width - 1, pt3[0] + 6)

    return pt1, pt2, pt3, pt4

def my_infer(context,input_img, batch_size):
    # start engine
    engine = context.get_engine()
    assert (engine.get_nb_bindings() == 2)
    # create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * batch_size
    # convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype=np.float32)
    # alocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    return output

def rank_boxes(boxes):
    def getKey(item):
        return item[1] #sort by y1
    sorted_boxes = sorted(boxes,key=getKey)
    return sorted_boxes

def ndarray_sort(arr1):
    result_list=[]
    for arr in arr1:
        temp=[]
        for ss in arr:
            temp.append(ss[0])
            temp.append(ss[1])
        result_list.append(temp)
    result_list = rank_boxes(result_list)
    array_result = np.array(result_list).reshape(-1,4,2)
    return array_result

#init the tf models
def init_models(use_divice,model_file):
    if use_divice == "GPU":
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.TEST_GPU_ID
    # load model
    uff_model = open(model_file, 'rb').read()
    parser = uffparser.create_uff_parser()
    parser.register_input("input_images", (3, 768, 768), 0)
    parser.register_output("feature_fusion/concat_3")
    # create inference engine and context (aka session)
    trt_logger = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
    engine = trt.utils.uff_to_trt_engine(logger=trt_logger,
                                         stream=uff_model,
                                         parser=parser,
                                         max_batch_size=1, # 1 sample at a time
                                         max_workspace_size= 1 << 20, # 1 GB GPU memory workspace
                                         datatype=trt.infer.DataType.FLOAT) # that's very cool, you can set precision
    context = engine.create_execution_context()
    return context


def text_detect(context, im):
    text_lines = []
    im_new = im[:, :, ::-1]
    im_resized, (ratio_h, ratio_w) = resize_image(im_new)
    im_resized = im_resized.astype(np.float32)
    im_resized -= (123.68, 116.78, 103.94)
    processed_im = np.transpose(im_resized, axes=(2, 0, 1))
    processed_im = processed_im.copy(order='C')
    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()
    batch_size = 1
    output = my_infer(context, processed_im, batch_size)
    # return predictions
    output = output.reshape((1, 192, 192, 6))
    score = output[:, :, :, 0]
    geometry = output[:, :, :, 1:6]
    timer['net'] = time.time() - start

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

    if boxes is not None:
        score_boxes = boxes[:, 8]
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    # save to file
    if boxes is not None:
        index_score = 0
        for box in boxes:
            pt1 = []
            pt2 = []
            pt3 = []
            pt4 = []
            temp_lines = []
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            pt1.append(int(box[0, 0].tolist()))
            pt1.append(int(box[0, 1].tolist()))
            pt2.append(int(box[1, 0].tolist()))
            pt2.append(int(box[1, 1].tolist()))
            pt3.append(int(box[2, 0].tolist()))
            pt3.append(int(box[2, 1].tolist()))
            pt4.append(int(box[3, 0].tolist()))
            pt4.append(int(box[3, 1].tolist()))
            if pt1[0] >= 0 and pt1[1] >= 0 and pt2[0] >= 0 and pt2[1] >= 0 and pt3[0] >= 0 and pt3[
                1] >= 0 and pt4[0] >= 0 and pt4[1] >= 0:
                my_dict = {}
                my_temp_list = []
                pt1, pt2, pt3, pt4 = regular_pts(pt1, pt2, pt3, pt4, im.shape[1])
                my_temp_list.append(pt1)
                my_temp_list.append(pt2)
                my_temp_list.append(pt3)
                my_temp_list.append(pt4)

                my_dict["pts"] = my_temp_list
                my_dict["score"] = float(score_boxes[index_score].tolist())
                index_score = index_score + 1

                text_lines.append(my_dict)

    return text_lines


def dump_result(text_lines):
    text_detect_result = dict()
    text_detect_result['detections'] = text_lines

    return text_detect_result

def resize_image(im, max_side_len=768):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, channel = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32

    if resize_h <= 0:
        resize_h = 32
    if resize_w <= 0:
        resize_w = 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    result = np.zeros([768, 768, 3])
    result[:im.shape[0],:im.shape[1],:im.shape[2]] = im

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return result, (ratio_h, ratio_w)

def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 3:
        score_map = score_map[0, :, :]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    context = init_models(configs['use_device'].upper(),configs['model_files']['freeze74742.uff'])
    return {"context": context,
            "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model_detect, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model_detect, reqs):
    #print("model_detect_type"+str(type(model_detect)))
    context = model_detect["context"]
    batch_size = model_detect['batch_size']
    CTX.logger.info("inference begin ...")

    try:
        imges_no_type = pre_eval(batch_size, reqs)
        output = eval(context, imges_no_type)
        ret = post_eval(output, reqs)

    except ErrorImageTooSmall as e:
        message = "image is too small, should be more than 32x32"
        return [], e.code, str(message)

    except ErrorImageNdim as e:
        message = "image with invalid ndim, should be 3"
        return [], e.code, str(message)

    except ErrorImageTooLarge as e:
        message = "image is too large, should be in 4999x4999, less than 10MB"
        return [], e.code, str(message)
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
        if img.shape[2] == 4:
            img = img[:, :, :3]
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
        res_list = []
        if len(text_bboxes) == 0:
            CTX.logger.info("no text detected")
            resps.append({"code": 0, "message": "", "result": {}})
            continue
        result = dump_result(text_bboxes)
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(context, imges_no_type):
    output = []
    _t1 = time.time()
    for i in range(len(imges_no_type)):
        text_bboxes = text_detect(
            context, imges_no_type[i])
        output.append((text_bboxes))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output

