# -*- coding: utf-8 -*-
import traceback
import caffe
import cv2
import numpy as np
from evals.utils.error import *
from util_infe import *


def det_create_net(configs):
    det_model = dict()
    det_code = 0
    det_message = ""
    deploy = str(configs['model_files']["det_deploy.prototxt"])
    weight = str(configs['model_files']["det_weight.caffemodel"])
    labelfile = str(configs['model_files']["det_labels.csv"])
    batch_size = 1
    if "batch_size" in configs:
        batch_size = configs['batch_size']
    try:
        change_deploy(deploy_file=deploy, input_data_batch_size=batch_size)
        net = caffe.Net(deploy, weight, caffe.TEST)
        label_dict = parse_label_file(labelfile)
        det_model['net'] = net
        det_model['label'] = label_dict
    except Exception as e:
        det_code = 400
        det_message = "create model error"
        det_model = {}
    return det_model, det_code, det_message


def det_net_inference(det_model, images):
    try:
        resized_images, images_h_w = det_pre_eval(images)
        output = det_eval(det_model['net'], resized_images)
        ret = det_post_eval(images_h_w, output, det_model['label'])
    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)
    return ret, 0, ''


def det_pre_eval(images):
    resized_images = []
    images_h_w = []
    for index, i_data in enumerate(images):
        height, width, _ = i_data.shape
        images_h_w.append([height, width])
        resized_images.append(det_preProcessImage(oriImage=i_data))
    return resized_images, images_h_w


def det_post_eval(images_h_w, output, label_dict):
    resps = []
    _t1 = time.time()
    # output_bbox_list : bbox_count * 7
    output_bbox_list = output['detection_out'][0][0]
    image_result_dict = dict()  # image_id : bbox_list
    for i_bbox in output_bbox_list:
        # i_bbox : length == 7 ; 0==image_id,1==class_index,2==score,3==bbox_xmin,4==bbox_ymin,5==bbox_xmax,6==bbox_ymax
        image_id = int(i_bbox[0])
        if image_id >= len(images_h_w):
            break
        h = images_h_w[image_id][0]
        w = images_h_w[image_id][1]
        class_index = int(i_bbox[1])
        if class_index < 1:  # background index == 0 , refinedet not output background info ,so the line not used
            continue
        score = float(i_bbox[2])
        if score < float(label_dict['threshold'][class_index]):
            continue
        name = label_dict['class'][class_index]
        bbox_dict = dict()
        bbox_dict['index'] = class_index
        bbox_dict['score'] = score
        bbox_dict['class'] = name
        bbox = i_bbox[3:7] * np.array([w, h, w, h])
        bbox_dict['pts'] = []
        xmin = int(bbox[0]) if int(bbox[0]) > 0 else 0
        ymin = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xmax = int(bbox[2]) if int(bbox[2]) < w else w
        ymax = int(bbox[3]) if int(bbox[3]) < h else h
        bbox_dict['pts'].append([xmin, ymin])
        bbox_dict['pts'].append([xmax, ymin])
        bbox_dict['pts'].append([xmax, ymax])
        bbox_dict['pts'].append([xmin, ymax])
        if image_id not in image_result_dict:
            image_result_dict[image_id] = []
        image_result_dict.get(image_id).append(bbox_dict)
    for image_id in range(len(images_h_w)):
        if image_id in image_result_dict:
            resps.append(image_result_dict[image_id])
        else:
            resps.append([])  # the image_id output zero bbox info
    return resps


def det_preProcessImage(oriImage=None):
    img = cv2.resize(oriImage, (320, 320))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    img = img.transpose((2, 0, 1))
    return img


def det_eval(net, images):
    '''
        eval forward inference
        Return
        ---------
        output: network numpy.mdarray
    '''
    for index, i_data in enumerate(images):
            net.blobs['data'].data[index] = i_data
    output = net.forward()
    if 'detection_out' not in output or len(output['detection_out']) < 1:
        raise ErrorForwardInference()
    return output


def det_checkSendToDetectOrNot(det_result, det_label_clsNeed):
    # det_model_need_index = []
    # for key in det_model['label']['clsNeed']:
    #     if int(det_model['label']['clsNeed'][key]) == 0:
    #         det_model_need_index.append(int(key))
    retFlag = 0  # not send to detect model
    for bbox in det_result:
        if bbox['index'] in det_label_clsNeed:
            retFlag = 1
    return retFlag
