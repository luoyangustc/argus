#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
    Load Image
'''

import cv2
import os
import numpy as np

from evals.utils.error import ErrorFileNotExist, ErrorCV2ImageRead, ErrorInvalidPTS, ErrorImageTooLarge, ErrorImageTooSmall, ErrorImageNdim


def load_image(image_file, body=None):
    '''
        load whole image, 0~255 BGR numpy
    '''
    image = None
    if body is not None:
        if len(body) > 10 * 1024 * 1024:
            raise ErrorImageTooLarge(image_file)
        image = cv2.imdecode(np.asarray(bytearray(body), dtype=np.uint8), 1)
    else:
        if not os.path.exists(image_file):
            raise ErrorFileNotExist(image_file)
        if os.path.getsize(image_file) > 10 * 1024 * 1024:
            raise ErrorImageTooLarge(image_file)
        image = cv2.imread(image_file)
    if image is None:
        raise ErrorCV2ImageRead(image_file)
    if image.shape[0] <= 32 or image.shape[1] <= 32:
        raise ErrorImageTooSmall(image)
    if image.ndim != 3:
        raise ErrorImageNdim(image)
    if image.shape[0] > 4999 or image.shape[1] > 4999:
        raise ErrorImageTooLarge(image_file)
    return image


def load_image_roi(image_file, roi, roi_scale=1.0, output_square=0, body=None):
    '''
    Get a sub-image from ROI of an image
    Parameters
    ----------
    image_file: input image file
    roi: input ROI, in form of a list [x, y, w, h, ...] or a tuple (x, y, w, h, ...)
    roi_scale: the factor to scale the ROI before crop
    output_square: if set 1, output squared images
    Return: an image, type: numpy.mdarray
    '''

    image = load_image(image_file, body=body)

    cx = int(roi[0] + roi[2] / 2.0)
    cy = int(roi[1] + roi[3] / 2.0)
    wd = int(roi[2] * roi_scale)
    ht = int(roi[3] * roi_scale)

    if output_square:
        wd = max(wd, ht)
        ht = wd

    roi_img = np.zeros((ht, wd, image.shape[2]), image.dtype)

    x0 = cx - wd / 2
    y0 = cy - ht / 2

    x1 = cx + wd / 2
    y1 = cy + ht / 2

    dx0 = 0
    dy0 = 0

    if x0 < 0:
        dx0 = -x0
        x0 = 0

    if y0 < 0:
        dy0 = -y0
        y0 = 0

    x1 = min(image.shape[1] - 1, x1)
    y1 = min(image.shape[0] - 1, y1)

    wd = x1 - x0
    ht = y1 - y0

    roi_img[dy0:dy0 + ht, dx0:dx0 + wd] = image[y0:y1, x0:x1]
    return roi_img


def check_valid_pts(pts):
    return (len(pts) == 4 and
            pts[1][0] > pts[0][0] and
            pts[3][1] > pts[0][1] and
            pts[1][0] - pts[0][0] == pts[2][0] - pts[3][0] and
            pts[2][1] - pts[1][1] == pts[3][1] - pts[0][1])


def load_image_roi_by_4pts(image_file, pts, roi_scale=1.0, output_square=0, body=None):
    '''
    Get a list of sub-images from ROIs of an image
    Parameters
    ----------
    image_file: input image file
    pts: list or tuple of input ROIs,
            each ROI is in form of a list [[xl, yt],[xr, yt], [xr, yb], [xl, yb], ...]
            or a tuple ([xl, yt],[xr, yt], [xr, yb], [xl, yb], ...)
    roi_scale: the factor to scale the ROI before crop
    output_square: if set 1, output squared images
    Return: list of images, image type: numpy.mdarray
    '''

    if not check_valid_pts(pts):
        raise ErrorInvalidPTS(pts)

    roi = [pts[0][0], pts[0][1], pts[2][0] - pts[0][0], pts[2][1] - pts[0][1]]
    roi_img = load_image_roi(image_file, roi, roi_scale, output_square, body=body)

    return roi_img
