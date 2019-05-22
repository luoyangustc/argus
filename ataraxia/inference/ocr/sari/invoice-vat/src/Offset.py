# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.decomposition import FastICA
from MaskGenerator import LineMaskGenerator
from evals.src.utils import findContours

def offset(img, pos, resize):
    '''
    根据感兴趣区域计算偏移量
    :param img: 输入图片
    :param pos: 图片中感兴趣区域
    :param resize: 图片放缩比率
    :return: dx,dy,cx,cy为中心偏移和角度偏转
    '''
    left = int((pos[0][0]) * resize)
    top = int((pos[0][1]) * resize)
    right = int((pos[1][0]) * resize)
    bottom = int((pos[1][1]) * resize)
    img = 255 - img[top:bottom, left:right]
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    canvas = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.uint8)
    canvas[...] = 255
    canvas[int(img.shape[0] * 1 / 2):int(img.shape[0] * 3 / 2),
           int(img.shape[1] * 1 / 2):int(img.shape[1] * 3 / 2)] = img.copy()
    erode = canvas
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 11))
    erode = cv2.morphologyEx(erode, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 55))
    erode = cv2.morphologyEx(erode, cv2.MORPH_OPEN, element)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    erode = cv2.dilate(erode, kernel)

    res = erode[int(img.shape[0] * 1 / 2):int(img.shape[0] * 3 / 2),
                int(img.shape[1] * 1 / 2):int(img.shape[1] * 3 / 2)]
    contours, hierarchy = findContours(
        res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(res, [box], 0, (128, 0, 0), 2)
    cv2.drawContours(img, [box], 0, (128, 0, 0), 2)

    dx1, dy1 = (box[2, 0] - box[1, 0], box[2, 1] - box[1, 1])
    dx2, dy2 = (box[1, 0] - box[0, 0], box[1, 1] - box[0, 1])

    cx = int((box[2, 0] + box[1, 0] + box[0, 0] + box[3, 0]) / 4)
    cy = int((box[2, 1] + box[1, 1] + box[0, 1] + box[3, 1]) / 4)
    dx = 0
    dy = 0
    if abs(dx2) > abs(dx1):
        dx = dx2
        dy = dy2
    else:
        dx = dx1
        dy = dy1
    if dx < 0:
        dx = -dx
        dy = -dy
    return dx, dy, cx, cy