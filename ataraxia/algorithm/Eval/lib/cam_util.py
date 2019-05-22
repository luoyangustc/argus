#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import logging
import cv2
import mxnet as mx
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gen_det_map(conv_feat_map, fc_weights):
    '''
    '''
    assert len(fc_weights.shape) == 2
    if len(conv_feat_map.shape) == 3:
        C, H, W = conv_feat_map.shape
        assert fc_weights.shape[1] == C
        detection_map = fc_weights.dot(conv_feat_map.reshape(C, H * W))
        detection_map = detection_map.reshape(-1, H, W)
    elif len(conv_feat_map.shape) == 4:
        N, C, H, W = conv_feat_map.shape
        assert fc_weights.shape[1] == C
        M = fc_weights.shape[0]
        detection_map = np.zeros((N, M, H, W))
        for i in xrange(N):
            tmp_detection_map = fc_weights.dot(
                conv_feat_map[i].reshape(C, H * W))
            detection_map[i, :, :, :] = tmp_detection_map.reshape(-1, H, W)
    return detection_map


def target_search(origin_mat, target_shape, top_k=5):
    '''
    :params:
    origin_mat      the whole original matrix(ndarray)
    target_shape    shape of target matrix
    top_k:          set the top k highest activation to choose from
                    0 means naive sliding-window search 

    :return:
    target_index    coordinates of top-left point of max window, (x, y) 
    '''
    def topleft_coordinates(center_idx, edge_len):
        if edge_len%2 == 0:     # (8,) -> (3,)
            return max(center_idx-(edge_len/2-1),0)
        elif edge_len%2 == 1:   # (7,) -> (3,)
            return max(center_idx-int(edge_len/2),0)
        else:
            logging.error("Invalid param: edge_len")
            return None

    assert len(origin_mat.shape)==len(target_shape)==2, logging.error('Only 2-D matrix is supported for now')
    assert target_shape[0]<=origin_mat.shape[0] and target_shape[1]<=origin_mat.shape[1], logging.error('Target matrix should be smaller than original matrix')
    target_w, target_h = target_shape
    target_index = 0, 0
    max_mean = 0
    if top_k == 0:  # sliding-window search
        for i in range(origin_mat.shape[0]-target_w+1):
            for j in range(origin_mat.shape[1]-target_h+1):
                cur = origin_mat[i:i+target_w, j:j+target_h]
                if cur.mean() > max_mean:
                    target_index = i, j
                    max_mean = cur.mean()
    elif isinstance(top_k, int):    # only search in top-k-max items
        dup_origin_mat = origin_mat.copy()
        for i in range(top_k):
            cur_center_ind = np.unravel_index(np.argmax(dup_origin_mat), dup_origin_mat.shape)
            cur_topleft_ind = topleft_coordinates(cur_center_ind[0], target_w), topleft_coordinates(cur_center_ind[1], target_h)
            cur = origin_mat[cur_topleft_ind[0]:cur_topleft_ind[0]+target_w, cur_topleft_ind[1]:cur_topleft_ind[1]+target_h]
            if cur.mean() > max_mean:
                target_index = cur_topleft_ind
                max_mean = cur.mean()
            dup_origin_mat[cur_center_ind] = dup_origin_mat.min()
    else:
        logging.error("Invalid param: top_k")
        return None
    return target_index 
        

def recover_coordinates(img, sample_rate, target_shape, target_index):
    '''
    '''
    img_h, img_w = img.shape[:2]
    rec_x, rec_y = int(target_index[0]*sample_rate[0]), int(target_index[1]*sample_rate[1])
    rec_tl = img[rec_x:int(rec_x+target_shape[0]*sample_rate[0]), rec_y:int(rec_y+target_shape[1]*sample_rate[1])]
    return rec_tl
    

def draw_cam(output_path, rgb_img, width, height, top_k, conv_fm, fc_weights, category, score, display=False):
    '''
    draw class active map
    '''
    score_sorted = -np.sort(-score)[:top_k]
    idx_sorted = np.argsort(-score)[:top_k]
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 1 + top_k, 1)
    plt.imshow(rgb_img)
    cam = gen_det_map(conv_fm, fc_weights[idx_sorted, :])
    for k in xrange(top_k):
        detection_map = np.squeeze(cam.astype(np.float32)[k, :, :])
        heat_map = cv2.resize(detection_map, (width, height))
        max_response = detection_map.mean()
        heat_map /= heat_map.max()
        im_show = rgb_img.astype(np.float32)/255*0.3 + plt.cm.jet(heat_map/heat_map.max())[:,:,:3]*0.7
        plt.subplot(1, 1 + top_k, k + 2)
        plt.imshow(im_show)
        print('Top {}: {}({:.6f}), max_response={:.4f}'.format(k + 1, category[idx_sorted[k]], score_sorted[k], max_response))
    if display:
        plt.show()
    plt.savefig(output_path)
    plt.close()
    return
