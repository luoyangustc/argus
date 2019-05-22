# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys
import evals.src.utils as utils


def detect_daikai_stamp(img):

    img_dict = {'_DaiKaiJiGuanGaiZhang': []}
    coord_dict  = {'_DaiKaiJiGuanGaiZhang': []}

    img1 = img.copy()
    start_y = np.min(np.where(img1 != (0, 0, 0))[0])
    end_y = np.max(np.where(img1 != (0, 0, 0))[0])
    img1[start_y:min(img.shape[0], start_y + int((end_y-start_y)*1/3)), :] = 255
    img1[max(0, end_y-int((end_y-start_y)*1/3)):end_y, :] = 255

    img1[np.where(img1 == (0,0,0))] = np.average(img1[np.where(img1 != (0,0,0))])
    img_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    bi_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    h[np.where(bi_img == 255)] = 0
    s[np.where(bi_img == 255)] = 0
    v[np.where(bi_img == 255)] = 0

    #cond = (h > 200) | (h <= 10) & (v > 46)
    cond = (h > 150)

    stamp_mask = np.zeros_like(h)
    stamp_mask[np.where(cond)] = 255

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    stamp_mask = cv2.erode(stamp_mask, element)

    stamp_mask = cv2.GaussianBlur(stamp_mask, (35,35), 0)
    contours, hierarchy = utils.findContours(stamp_mask)

    boxes = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
        #cv2.drawContours(stamp_mask, [box], 0, 128, 2)

    max_area = 0
    idx = -1
    for i, box in enumerate(boxes):
        area = cv2.contourArea(box)
        if area > max_area:
            idx = i
            max_area = area

    if idx != -1:
        #cv2.drawContours(stamp_mask, boxes, idx, 128, 5)
        ratio = cv2.contourArea(boxes[idx])/(img.shape[0]*(end_y-start_y))
        if ratio > 0.05 and ratio < 0.19:
            left = np.min(boxes[idx][:,0])
            right = np.max(boxes[idx][:,0])
            top = np.min(boxes[idx][:, 1])
            bottom = np.max(boxes[idx][:, 1])
            img_dict['_DaiKaiJiGuanGaiZhang'] = img[top:bottom, left:right].copy()
            coord_dict['_DaiKaiJiGuanGaiZhang'] = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
            #utils.show('stamp', img_dict['_DaiKaiJiGuanGaiZhang'], True)


    return img_dict, coord_dict