# -*- coding:utf-8 -*-  
import cv2
import numpy as np
import os
from io import open


class Matcher(object):
    '''
        功能：用于发票对齐
        备注：利用SIFT描述子对齐发票
    '''
    def __init__(self):
        '''
        加载sift特征提取器,匹配算法参数
        '''
        # self.sift = cv2.SIFT()
        self.sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
        #search_params = dict(checks=10)
    #FLANN_INDEX_LSH = 4
        index_params= dict(algorithm = 4,table_number = 6, key_size = 12,multi_probe_level = 1)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def sift_fet(self, img, input_roi=None):
        '''
        提取sift特征
        :param img:输入图片
        :param input_roi:图片中的部分区域，为None时为全图
        :return:sift特征
        '''
        if input_roi != None:
            _img = img[input_roi[1]:input_roi[3], input_roi[0]:input_roi[2]]
        else:
            _img = img
        return self.sift.detectAndCompute(_img, None)

    def post_match(self, kp1, des1, kp2, des2, color_img, threshold=0.7, good_num=10, input_roi=None, outputsize=None,
                   offset=None, resize=1):
        '''
        发票对齐
        :param kp1: 模板点
        :param des1:模板特征
        :param kp2:待对齐点
        :param des2:待对齐特征
        :param color_img:待对齐原图
        :param threshold:匹配阈值
        :param good_num:匹配最小个数
        :param input_roi:待对齐图感兴趣区域
        :param outputsize:输出图片大小
        :param offset:对齐后的偏移
        :param resize:待对齐图与原图的比率
        :return:对齐发票
        '''
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append(m)
        if len(good) > good_num:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
            if input_roi != None:
                dst_pts = np.float32(dst_pts + [input_roi[0], input_roi[1]])
            dst_pts = dst_pts.reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if offset != None:
                M = np.dot(np.float32(
                    [[1, 0, -offset[0]], [0, 1, -offset[1]], [0, 0, 1]]), M)
            matchesMask = mask.ravel().tolist()
            M = np.dot(np.float32([[1.0 / resize, 0, 0], [0, 1.0 / resize, 0], [0, 0, 1]]),
                       np.dot(M, np.float32([[resize, 0, 0], [0, resize, 0], [0, 0, 1]])))  # *2
            if outputsize == None:
                outputsize = (color_img.shape[1], color_img.shape[0])
            return cv2.warpPerspective(color_img, M,
                                       outputsize), np.linalg.inv(M)  # M
        else:
            return None, None  # M
