# -*- coding: utf-8 -*-
import traceback
import caffe
import cv2
import csv
import numpy as np
from evals.utils.error import *
from util_infe import *
from collections import OrderedDict, defaultdict


def get_label_map(labelmap):
    map_dict = {}
    with open(labelmap, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            map_dict[int(line[0])] = int(line[1])
    return map_dict


def cls_create_net(configs):
    cls_model = dict()
    cls_code = 0
    cls_message = ""

    fine_deploy = str(configs['model_files']["fine_deploy.prototxt"])
    fine_weight = str(configs['model_files']["fine_weight.caffemodel"])
    fine_labels = str(configs['model_files']["fine_labels.csv"])
    percent_fine = float(configs['custom_params']['percentage_fine'])
    coarse_deploy = str(configs['model_files']['coarse_deploy.prototxt'])
    coarse_weight = str(configs['model_files']['coarse_weight.caffemodel'])
    coarse_labels = str(configs['model_files']['coarse_labels.csv'])
    percent_coarse = float(configs['custom_params']['percentage_coarse'])
    if "batch_size" in configs:
        batch_size = configs['batch_size']
    else:
        batch_size = 1
    try:
        change_deploy(deploy_file=fine_deploy,
                      input_data_batch_size=batch_size)
        change_deploy(deploy_file=coarse_deploy,
                      input_data_batch_size=batch_size)
        net_fine = caffe.Net(fine_deploy, fine_weight, caffe.TEST)
        net_coarse = caffe.Net(coarse_deploy, coarse_weight, caffe.TEST)

        cls_model = {"net_fine": net_fine, "net_coarse": net_coarse, "percentage_fine": percent_fine,
                     "percentage_coarse": percent_coarse,
                     "dict_label": parse_label_file(fine_labels),
                     "label_map_dict": get_label_map(coarse_labels)}
    except Exception as e:
        cls_code = 400
        cls_message = "create model error :" + str(e)
        cls_model = {}
    return cls_model, cls_code, cls_message


def cls_net_inference(cls_model, images):
    try:
        resized_images = cls_pre_eval(images)
        output_fine = cls_eval(cls_model['net_fine'], resized_images)
        output_coarse = cls_eval(cls_model['net_coarse'], resized_images)
        cur_batchsize = len(resized_images)
        ret = cls_post_eval(output_fine, output_coarse,
                            cur_batchsize, cls_model)
        #ret_final = score_map(ret, cur_batchsize, model)
    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)
    return ret, 0, ''


def cls_pre_eval(images):
    img = []
    for image in images:
        img.append(preProcessImage(image))
    return img


def center_crop(img, crop_size):
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy:yy + crop_size, xx: xx + crop_size]


def preProcessImage(img=None):
    img = img.astype(np.float32, copy=True)
    img = cv2.resize(img, (256, 256))
    # img = img.astype(np.float32, copy=True)
    img -= np.array([[[103.94, 116.78, 123.68]]])
    img = img * 0.017
    img = center_crop(img, 225)
    img = img.transpose((2, 0, 1))
    return img


def cls_eval(net, images):
    '''
        eval forward inference
    '''
    for index, i_data in enumerate(images):
            net.blobs['data'].data[index] = i_data
    output = net.forward()
    if 'prob' not in output or len(output['prob']) < len(images):
        raise ErrorForwardInference()
    return output


def cls_post_eval(output_fine, output_coarse, cur_batchsize, cls_model):
    '''
        parse net output, as numpy.mdarray, to EvalResponse
    '''
    resps = []

    percent_fine = cls_model['percentage_fine']
    percent_coarse = cls_model['percentage_coarse']
    dict_label = cls_model['dict_label']
    map_dict = cls_model['label_map_dict']

    coarse_label_to_fine_label = defaultdict(list)
    for key in map_dict.keys():
        coarse_label_to_fine_label[map_dict[key]].append(key)

    num_labels = len(dict_label['class'])

    for i in range(cur_batchsize):
        output_prob_fine = np.squeeze(output_fine['prob'][i])
        output_prob_coarse = np.squeeze(output_coarse['prob'][i])
        merge_conf = cls_merge_fine_coarse(
            output_prob_fine, output_prob_coarse, coarse_label_to_fine_label, percent_fine, percent_coarse)
        confidences = cls_score_map(merge_conf, dict_label)
        resps.append(confidences)
    return resps


def get_max_idx(array,  index_list):
    max_idx = index_list[0]
    max_value = array[max_idx]
    for i in index_list:
        if array[i] > max_value:
            max_idx = i
            max_value = array[i]
    return max_idx


def cls_merge_fine_coarse(output_prob_fine, output_prob_coarse, coarse_label_to_fine_label, percent_fine, percent_coarse):
    new_conf = [score for score in output_prob_fine]
    for key in coarse_label_to_fine_label.keys():
        max_idx = get_max_idx(
            output_prob_fine, coarse_label_to_fine_label[key])
        new_conf[max_idx] = output_prob_coarse[key] * \
            percent_coarse + output_prob_fine[max_idx] * percent_fine

    return new_conf

def cls_score_map(merge_conf, dict_label):
    score_map_conf = []
    for index in range(len(merge_conf)):
        model_threshold = float(dict_label['model_threshold'][index])
        serving_threshold = float(dict_label['serving_threshold'][index])

        score = merge_conf[index]
        if score < model_threshold:
            score_map = score * serving_threshold / model_threshold
        else:
            score_map = 1 - (1-score) * (1-serving_threshold) / \
                (1-model_threshold)
        score_map_conf.append(score_map)
    return score_map_conf


def merge_confidences(cls_result, cls_model):
    dict_label = cls_model['dict_label']
    cls_result_merge = []
    # Handling the results of each graph in a multi-batch
    for single_img_res in cls_result:
        # find label index which name is same
        label_index = OrderedDict()
        num_label = len(dict_label['class'])
        for key in range(num_label):
            label = dict_label['class'][key]
            label_index.setdefault(label, []).append(key)
        # choose the max score for the same label
        result = []
        for key in label_index.keys():
            max_score = max(
                float(single_img_res[i]) for i in label_index[key])
            index = int(get_max_idx(single_img_res, label_index[key]))
            res = {"index": index, "class": key, "score": max_score}
            result.append(res)
        cls_result_merge.append({'confidences': result})
    return cls_result_merge


def cls_merge_det(cls_result, det_result, cls_model, det_model):
    dict_label = cls_model['dict_label']
    cls_need = det_model["label"]["clsNeed"]
    # get the index of detection label that need merge det result with cls result.
    need_merge_det_idx = []
    for key in cls_need.keys():
        if 'yes' in cls_need[key]:
            need_merge_det_idx.append(int(key))
    # Handling the results of each graph in a multi-batch
    for i in range(len(cls_result)):
        det_res = det_result[i]
        # get cls label that need change score, determine by
        # whether there is a detection result of this category.
        det_find_label_idx = []
        for res in det_res:
            if int(res['index']) in need_merge_det_idx:
                det_find_label_idx.append(int(res['index']))

        det_not_find_label_idx = list(
            set(need_merge_det_idx) - set(det_find_label_idx))

        for idx in det_not_find_label_idx:
            # category mapping
            for cls_index in cls_need[idx].split('_')[1:]:
                cls_index = int(cls_index)
                serving_threshold = float(
                    dict_label['serving_threshold'][cls_index])
                label = dict_label['class'][cls_index]
                score = cls_result[i][cls_index]
                if score >= serving_threshold:
                    if label != 'normal':
                        score_map = serving_threshold - 0.01
                        cls_result[i][cls_index] = score_map
    return cls_result
