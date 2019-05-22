# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib
import utils


def extractFeature(im):
    # 背景纯净化
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_im[np.where(gray_im == 255)] = np.average(gray_im[np.where(gray_im != 255)])
    bi_mask = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bi_mask = 255 - bi_mask

    valid_num0 = im[np.where(bi_mask == 255)].shape[0]

    # 章
    im[np.where(bi_mask == 0)] = 255
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h = im_hsv[:, :, 0]
    v = im_hsv[:, :, 2]
    stamp_mask = np.zeros_like(bi_mask)

    cond = (h >= 150) | (h <= 10) & (v > 150)
    stamp_mask[np.where(cond)] = 255
    stamp_mask = 255-stamp_mask
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stamp_mask = cv2.erode(stamp_mask, element)

    valid_num = im[np.where(stamp_mask == 255)].shape[0]

    blured_mask = cv2.GaussianBlur(stamp_mask, (15,15), 0)
    contours, hierarchy = utils.findContours(blured_mask)

    min_y = im.shape[0]/2
    max_y = im.shape[0]/2
    min_x = im.shape[1]/2
    max_x = im.shape[1]/2
    for cnt in contours:
        min_x = min(min_x, np.min(cnt[:, :, 0]))
        max_x = max(max_x, np.max(cnt[:, :, 0]))
        min_y = min(min_y, np.min(cnt[:, :, 1]))
        max_y = max(max_y, np.max(cnt[:, :, 1]))

    min_x = min_x * 1.0 / im.shape[1]
    max_y = max_y*1.0/im.shape[0]

    # 计算contour平均面积
    avg_area = 0
    for cnt in contours:
        avg_area += cv2.contourArea(cnt)
    if len(contours) != 0:
        avg_area /= len(contours)
    else:
        avg_area = 0

    return valid_num0, valid_num, avg_area, min_x, max_y


class clsPredictor_FiaoPiaoLianCi():
    def __init__(self, model_path):
        self.clf = joblib.load(model_path)

    def predict(self, img):
        try:
            rets = extractFeature(img)
            feature = []
            for ret in rets:
                feature = np.append(feature, ret)

            predict = self.clf.predict([feature])
        except:
            predict = None
        return predict
