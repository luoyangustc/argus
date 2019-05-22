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
    array_result = np.array(result_list).reshape(-1, 4, 2)
    return array_result

#init the tf models
def init_models(use_divice,model_file):
    model_file_new = model_file[0:-22]
    if use_divice == "GPU":
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.TEST_GPU_ID
    # initialize the detectors
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        ckpt_state = tf.train.get_checkpoint_state(model_file_new)
        model_path = os.path.join(model_file_new, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

    return sess, input_images, f_score, f_geometry


def text_detect(text_detector, im, f_score, f_geometry, input_images):
    text_lines = []
    #print("text_detect:"+str(im.shape))
    im_new = im[:, :, ::-1]
    im_resized, (ratio_h, ratio_w) = resize_image(im_new)
    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()
    score, geometry = text_detector.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
    timer['net'] = time.time() - start

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
    #print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
    #    image_path, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
    #print("boxes:size"+str(len(boxes)))
    if boxes is not None:
        score_boxes = boxes[:, 8]
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    #duration = time.time() - start_time
    #print('[timing] {}'.format(duration))
    # save to file
    if boxes is not None:
        index_score = 0
        for box in boxes:
            temp_lines = []
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            t_00 = int(box[0, 0].tolist())
            t_01 = int(box[0, 1].tolist())
            t_10 = int(box[1, 0].tolist())
            t_11 = int(box[1, 1].tolist())
            t_20 = int(box[2, 0].tolist())
            t_21 = int(box[2, 1].tolist())
            t_30 = int(box[3, 0].tolist())
            t_31 = int(box[3, 1].tolist())
            if t_00 >= 0 and t_01 >= 0 and t_10 >= 0 and t_11 >= 0 and t_20 >= 0 and t_21 >= 0 and t_30 >= 0 and t_31 >= 0:
                my_dict = {}
                my_temp_list =[]
                pt1 = []
                pt2 = []
                pt3 = []
                pt4 = []
                pt1.append(t_00)
                pt1.append(t_01)
                pt2.append(t_10)
                pt2.append(t_11)
                pt3.append(t_20)
                pt3.append(t_21)
                pt4.append(t_30)
                pt4.append(t_31)
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
    h, w, _ = im.shape

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

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

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
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
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
    text_detector,input_images,f_score,f_geometry = init_models(configs['use_device'].upper(),configs['model_files']['model.ckpt-26262.index'])
    return {"text_sess_name": text_detector,
            "input_images":input_images,
            "f_score":f_score,
            "f_geometry":f_geometry,
            "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model_detect, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model_detect, reqs):
    #print("model_detect_type"+str(type(model_detect)))
    text_detector = model_detect["text_sess_name"]
    input_images = model_detect['input_images']
    f_score = model_detect['f_score']
    f_geometry = model_detect['f_geometry']
    batch_size = model_detect['batch_size']
    CTX.logger.info("inference begin ...")

    try:
        imges_no_type = pre_eval(batch_size, reqs)
        output = eval(text_detector, imges_no_type,f_score,f_geometry,input_images)
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
        #print(img.shape)
        ret.append((img))

    _t2 = time.time()
    CTX.logger.info("load: %f", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)
    #print("ret:"+str(ret[0]))
    #print("ret_size:"+str(ret[0].shape))
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


def eval(text_detector, imges_no_type,f_score,f_geometry,input_images):
    output = []
    _t1 = time.time()
    for i in range(len(imges_no_type)):
        text_bboxes = text_detect(
            text_detector, imges_no_type[i],f_score,f_geometry,input_images)
        #print(text_bboxes)
        output.append((text_bboxes))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output
#
# if __name__ == '__main__':
#     configs = {
#         "app": "bkapp",
#         "use_device": "GPU",
#         "batch_size":256
#     }
#     result_dict,_,_=create_net(configs)
#     net_preprocess(model, req)
#     print(type(result_dict))
#     img_path = "/workspace/imagenet-data/EAST/test_pic_0510/"
#     img_list = os.listdir(img_path)
#     reqs=[]
#     temp_i = 0
#     for img_name in img_list:
#         reqs_temp = dict()
#         reqs_temp["data"]=dict()
#         reqs_temp["data"]["uri"]=img_path + img_name
#         reqs_temp["data"]["body"]=None
#         reqs.append(reqs_temp)
#     ret = net_inference(result_dict, reqs)
#     print(ret)
