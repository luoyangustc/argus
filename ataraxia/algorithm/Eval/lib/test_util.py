#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import math
import time
import logging
import cv2
import mxnet as mx
import numpy as np
from collections import namedtuple
from io_hybrid import np_img_preprocessing,np_img_center_crop,np_img_multi_crop
from config import cfg
import multiprocessing
import functools
import random


def _get_filename_with_parents(filepath, level=1):
    common = filepath
    for i in range(level + 1):
        common = os.path.dirname(common)
    return os.path.relpath(filepath, common)


def _image_processor(img, ex):
    error_img = None
    try:
        img_read = cv2.imread(img)
        if np.shape(img_read) == tuple():
            raise empty_image
    except:
        img_read = np.zeros((ex['input_shape'][1], ex['input_shape'][2], ex['input_shape'][0]), dtype=np.uint8)
        if ex['base_name']:
            error_img = os.path.basename(img)
        else:
            error_img = _get_filename_with_parents(img, level=ex['level'])
        logging.error('Image error: {}, result will be deprecated!'.format(img))
    img_tmp = np_img_preprocessing(img_read, **ex['img_preproc_kwargs'])

    if ex['center_crop']:
        img_ccr = np_img_ex['center_crop'](img_tmp, ex['input_shape'][1]) 
        if np.__version__.startswith('1.15'):
            return mx.nd.array(img_ccr), error_img
        else:
            return mx.nd.array(img_ccr[np.newaxis, :]), error_img
    elif ex['multi_crop']:
        img_crs = np_img_ex['multi_crop'](img_tmp, ex['input_shape'][1], crop_number=ex['multi_crop'])
        for idx_crop,crop in enumerate(img_crs):
            if np.__version__.startswith('1.15'):
                return [mx.nd.array(crop) for crop in img_crs], error_img 
            else:
                return [mx.nd.array(crop[np.newaxis, :]) for crop in img_crs], error_img 
    else:
        if np.__version__.startswith('1.15'):
            return mx.nd.array(img_tmp), error_img
        else:
            return mx.nd.array(img_tmp[np.newaxis, :]), error_img


def infer_one_batch(model, categories, data_batch, img_list, base_name=True, multi_crop_ave=False):
    '''
    '''
    Batch = namedtuple('Batch', ['data'])
    results_one_batch = list() 
    k = cfg.TEST.TOP_K
    level = cfg.TEST.FNAME_PARENT_LEVEL 
    model.forward(Batch([data_batch]))
    output_prob_batch = model.get_outputs()[0].asnumpy()
    for idx, img_name in enumerate(img_list):
        if multi_crop_ave:
            # 3x3:[[cls_0],[cls_1],[cls_2]] -> 3x1:[cls_0_avg,cls_1_avg,cls_2_avg]
            output_prob = np.average(output_prob_batch,axis=0)
        else:
            output_prob = output_prob_batch[idx]
        
        # sort index-list and create sorted rate-list
        index_list = output_prob.argsort()
        rate_list = output_prob[index_list]
        _index_list = index_list.tolist()[-k:][::-1]
        _rate_list = rate_list.tolist()[-k:][::-1]

        # write result dictionary
        result = dict()
        if base_name:
            result['File Name'] = os.path.basename(img_name)
        else:
            result['File Name'] = _get_filename_with_parents(img_name, level=level)
        # result['File Name'] = img_name 

        # get top-k indices and revert to top-1 at first
        result['Top-{} Index'.format(k)] = _index_list
        result['Top-{} Class'.format(k)] = [categories[int(x)] for x in _index_list]

        # use str to avoid JSON serializable error
        result['Confidence'] = [str(x) for x in list(output_prob)] if cfg.TEST.LOG_ALL_CONFIDENCE else [str(x) for x in _rate_list]
        results_one_batch.append(result)
    return results_one_batch
    

def generic_multi_gpu_test(model, img_list, categories, batch_size, input_shape, img_preproc_kwargs, center_crop=False, multi_crop=None, h_flip=False, img_prefix=None, base_name=True):
    '''
    '''
    timer = 0
    level = cfg.TEST.FNAME_PARENT_LEVEL 
    result = dict()
    count = 0 
    err_num = 0
    multi_crop_ave = True if multi_crop else False
    img_num = len(img_list)
    proc_pool = multiprocessing.Pool(cfg.TEST.PROCESS_NUM)
    logging.info("Processing images with {} procs".format(cfg.TEST.PROCESS_NUM))
    while(img_list):
        count += 1
        # list of one batch data
        buff_list = list()
        error_list = list()
        buff_size = 1 if multi_crop else batch_size
        for i in range(buff_size):
            if not img_list:
                logging.debug("current list empty")
                break
            elif img_prefix:
                buff_list.append(img_prefix + img_list.pop(0))
            else:
                buff_list.append(img_list.pop(0))
        # process one data batch
        tic = time.time()
        img_batch = mx.nd.array(np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))) 
        # timers = [0 for x in range(3)]
        # tic_0 = time.time()
        # -------------- single proc --------------
        if cfg.TEST.PROCESS_NUM == 1:
            for idx,img in enumerate(buff_list):
                # tic_1 = time.time()
                try:
                    img_read = cv2.imread(img)
                    if np.shape(img_read) == tuple():
                        raise empty_image
                except:
                    img_read = np.zeros((input_shape[1], input_shape[2], input_shape[0]), dtype=np.uint8)
                    if base_name:
                        error_list.append(os.path.basename(img))
                    else:
                        error_list.append(_get_filename_with_parents(img, level=level))
                    logging.error('Image error: {}, result will be deprecated!'.format(img))
                # timers[0]+=(time.time()-tic_1)
                # tic_2 = time.time()
                img_tmp = np_img_preprocessing(img_read, **img_preproc_kwargs)
                # print('improc:',time.time()-tic_2)
                # timers[1]+=(time.time()-tic_2)
                logging.debug('img_tmp.shape:{}'.format(img_tmp.shape))
                # tic_3 = time.time()
                if center_crop:
                    img_ccr = np_img_center_crop(img_tmp, input_shape[1]) 
                    if np.__version__.startswith('1.15'):
                        img_batch[idx] = mx.nd.array(img_ccr)
                    else:
                        img_batch[idx] = mx.nd.array(img_ccr[np.newaxis, :])
                elif multi_crop:
                    img_crs = np_img_multi_crop(img_tmp, input_shape[1], crop_number=multi_crop)
                    for idx_crop,crop in enumerate(img_crs):
                        if np.__version__.startswith('1.15'):
                            img_batch[idx_crop] = mx.nd.array(crop)
                        else:
                            img_batch[idx_crop] = mx.nd.array(crop[np.newaxis, :])
                else:
                    if np.__version__.startswith('1.15'):
                        img_batch[idx] = mx.nd.array(img_tmp)
                    else:
                        img_batch[idx] = mx.nd.array(img_tmp[np.newaxis, :])
                # timers[2]+=(time.time()-tic_3)
            # print('batch_timer:',timers)
            # print('batch:',time.time()-tic_0)

        # -------------- multi proc ---------------
        elif cfg.TEST.PROCESS_NUM > 1:
            extra_args = {
                'input_shape': input_shape, 
                'img_preproc_kwargs': img_preproc_kwargs, 
                'level': level, 
                'base_name': base_name, 
                'center_crop': center_crop, 
                'multi_crop': multi_crop
            }
            # tic_1 = time.time()
            processed_tuples = proc_pool.map(functools.partial(_image_processor, ex=extra_args), buff_list)
            # timers[0]+=(time.time()-tic_1)
            # tic_2 = time.time()
            processed_imgs = [x[0] for x in processed_tuples]
            error_list = [x[1] for x in processed_tuples if x[1]]
            for idx,img in enumerate(processed_imgs):
                img_batch[idx] = img
            # timers[1]+=(time.time()-tic_2)
        # print('batch_timer:',timers)
        # -----------------------------------------
        # tic_4 = time.time()
        buff_result = infer_one_batch(model, categories, img_batch, buff_list, base_name=True, multi_crop_ave=multi_crop_ave)
        toc = time.time()
        # timers[2]+=(time.time()-tic_4)
        # print('batch_infer:',time.time()-tic_4)
        # print('FLAG:{},{},{}'.format(timers[0],timers[1],timers[2]))
        timer+=(toc-tic)
        for buff in buff_result:
            result[buff["File Name"]] = buff
        for img in error_list:
            del result[img]
        err_num += len(error_list)
        logging.info("Batch [{}]:\tgpu_number={}\tbatch_size={}\terror_number={}\tbatch_time={:.3f}s".format(count, len(cfg.TEST.GPU_IDX), len(buff_result), len(error_list),toc-tic))
    logging.info("Tocal error image number={}".format(err_num))
    logging.info("Average time per batch(with preprocessing)={:.3f}s".format(timer/count))
    logging.info("Average time per image(with preprocessing)={:.3f}s".format(timer/img_num))
    return result


def single_image_test(model, image_path, categories, input_shape, img_preproc_kwargs, center_crop=False, multi_crop=False, h_flip=False):
    '''
    '''
    Batch = namedtuple('Batch', ['data'])
    k = cfg.TEST.TOP_K
    level = cfg.TEST.FNAME_PARENT_LEVEL 
    multi_crop_ave = True if multi_crop else False
    try:
        img_read = cv2.imread(image_path)
        if np.shape(img_read) == tuple():
            raise empty_image
    except:
        logging.error('Reading image failed')
        return None
    logging.info('Shape of image after read-in: {}'.format(img_read.shape))
    img_tmp = np_img_preprocessing(img_read, **img_preproc_kwargs)
    logging.info('Shape of image after preprocessing: {}'.format(img_tmp.shape))

    # input data batch 
    if center_crop:
        img_ccr = np_img_center_crop(img_tmp, input_shape[1]) 
        if np.__version__.startswith('1.15'):
            img_batch = mx.nd.array(img_ccr)
        else:
            img_batch = mx.nd.array(img_ccr[np.newaxis, :])
    elif multi_crop:
        img_batch = mx.nd.array(np.zeros((multi_crop, input_shape[0], input_shape[1], input_shape[2])))
        img_crs = np_img_multi_crop(img_tmp, input_shape[1], crop_number=multi_crop)
        for idx_crop,crop in enumerate(img_crs):
            if np.__version__.startswith('1.15'):
                img_batch[idx_crop] = mx.nd.array(crop)
            else:
                img_batch[idx_crop] = mx.nd.array(crop[np.newaxis, :])
    else:
        if np.__version__.startswith('1.15'):
            img_batch = mx.nd.array(img_tmp)
        else:
            img_batch = mx.nd.array(img_tmp[np.newaxis, :])
    logging.info('Shape of data fed to model: {}'.format(img_batch.shape))

    # forward
    model.forward(Batch([img_batch]))
    # ==== debug ====
    # logging.debug(model.output_names)
    # for idx,layer in enumerate(model.output_names):
    #     layer_output = model.get_outputs()[idx].asnumpy()
    #     logging.debug('==> [{}]:{} {}\n{}'.format(idx, layer, layer_output.shape, layer_output[0,...]))
    #     if ("0" in layer or "bn_data" in layer) and "flatten" not in layer: 
    #         logging.debug('==> [{}]:{} {}\n{}'.format(idx, layer, layer_output.shape, layer_output[0,0,:10,:10]))
    # ===============
    output_prob_batch = model.get_outputs()[0].asnumpy()
    if multi_crop_ave:
        # 3x3:[[cls_0],[cls_1],[cls_2]] -> 3x1:[cls_0_avg,cls_1_avg,cls_2_avg]
        output_prob = np.average(output_prob_batch,axis=0)
    else:
        output_prob = output_prob_batch[0]
    
    # sort index-list and create sorted rate-list
    index_list = output_prob.argsort()
    rate_list = output_prob[index_list]
    _index_list = index_list.tolist()[-k:][::-1]
    _rate_list = rate_list.tolist()[-k:][::-1]

    # write result dictionary
    result = dict()
    result['File Name'] = os.path.basename(image_path)

    # get top-k indices and revert to top-1 at first
    result['Top-{} Index'.format(k)] = _index_list
    result['Top-{} Class'.format(k)] = [categories[int(x)] for x in _index_list]

    # use str to avoid JSON serializable error
    result['Confidence'] = [str(x) for x in list(output_prob)] if cfg.TEST.LOG_ALL_CONFIDENCE else [str(x) for x in _rate_list]
    return result


def mutable_images_test(model, image_list, categories, input_shape, img_preproc_kwargs, center_crop=False, multi_crop=None, h_flip=False, img_prefix=None, base_name=True):
    '''
    '''
    assert img_preproc_kwargs['keep_aspect_ratio'], logging.error('Mutable images testing should keep aspect ratio of input images')
    results = dict()
    for index,image in enumerate(image_list):
        if img_prefix:
            image = img_prefix + image 
        tic = time.time()
        _ = single_image_test(model, image, categories, input_shape, img_preproc_kwargs, center_crop=center_crop, h_flip=h_flip)
        logging.info("Batch [{}]:\tbatch_time={:.3f}s".format(index+1, time.time()-tic))
        if _:
            results[_['File Name']] = _ 
    return results


def test_wrapper(model, image_or_list, categories, batch_size, input_shape, kwargs, center_crop=False, multi_crop=None, h_flip=False, img_prefix=None, base_name=True, single_img_test=False, mutable_img_test=False):
    '''
    '''
    if single_img_test:
        return single_image_test(model, image_or_list, categories, input_shape, kwargs, center_crop=center_crop, multi_crop=multi_crop, h_flip=h_flip)
    elif mutable_img_test:
        return mutable_images_test(model, image_or_list, categories, input_shape, kwargs, center_crop=center_crop, h_flip=h_flip, img_prefix=img_prefix, base_name=base_name)
    else:
        return generic_multi_gpu_test(model, image_or_list, categories, batch_size, input_shape, kwargs, center_crop=center_crop, multi_crop=multi_crop, h_flip=h_flip, img_prefix=img_prefix, base_name=base_name) 
