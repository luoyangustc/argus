# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import numpy as np

if cv2.__version__[0] == '2':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
DEBUG = 0


class LineMaskGenerator(object):
    def __init__(self, k_x=45, k_y=45, min_len=50):
        self.k_x = k_x
        self.k_y = k_y
        self.min_len = min_len

    def eraseVerticalLine(self, binary_img):  # 保留横线

        (k_x, k_y) = (self.k_x, 1)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_x, k_y))
        eroded = cv2.erode(binary_img, element)
        dst = cv2.dilate(eroded, element)

        cv2.HoughLinesP(dst, 1, np.pi, 1,
                        minLineLength=self.min_len, maxLineGap=0)

        return dst

    def eraseHorizontalLine(self, binary_img):  # 保留竖线

        (k_x, k_y) = (1, self.k_y)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_x, k_y))
        eroded = cv2.erode(binary_img, element)
        dst = cv2.dilate(eroded, element)

        return dst

    def genMask(self, gray_img):
        """
        Gen Mask of an image
        :param gray_img: Gray image
        :return: A binary mask, where 255 represent detected horizontal or vertical line, 0 the others
        """
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        #gray_img = gray_img[:,:,2]

        ave = np.average(gray_img)
        if ave > 128:
            gray_img = 255 - gray_img

        # gray_img = (np.sqrt(gray_img / 255.0) * 255).astype(np.uint8)  # 强化后的灰度图
        #gray_img = ((gray_img) ** (2.0/3)).astype(np.uint8)
        gray_img = ((gray_img / 255.0) ** (2.0 / 3)
                    * 255 * 2.2 / 3).astype(np.uint8)
        _, binary_img = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img_v = self.eraseVerticalLine(binary_img)
        img_h = self.eraseHorizontalLine(binary_img)

        mask = img_v | img_h

        #mask = cv2.resize(mask, (gray_img.shape[1], gray_img.shape[0]))
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        return mask, binary_img - mask
