# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from Offset import offset
from MaskGenerator import LineMaskGenerator
import evals.src.utils
from io import open
from functools import cmp_to_key
from Machersolution import Matcher
from Finecut import Fine_cut
import DaiKaiStampDetection

class Coarse2Fine_cut(object):
    '''
    功能：用于发票字段切分
    备注：外部使用者主要调用cut函数将图像中的感兴趣字段进行切割
    '''
    '''
    输入：输入模板路径
    功能：加载对齐模板
          加载三类区域的坐标
    备注：zz3.jpg为标准模板图片，所有输入的图片均会参照它对齐
          result_step1.txt，印刷字段区域
          result_step2.txt，打印字段区域
          result_step3.txt，无关字段区域
    '''
    def __init__(self, template_root):
        '''
        功能：加载对齐模板
              加载三类区域的坐标
        备注：zz3.jpg为标准模板图片，所有输入的图片均会参照它对齐
              result_step1.txt，印刷字段区域
              result_step2.txt，打印字段区域
              result_step3.txt，无关字段区域
        :param template_root:模板路径
        '''
        # get template sift
        self.m = Matcher()
        template1 = cv2.resize(cv2.imread(os.path.join(template_root, 'zz3.jpg'), 0), None, None, 1, 1)
        self.kp1, self.des1 = self.m.sift_fet(template1)
        self.t1shape = template1.shape
        # get step 1 location
        annoated_file = open(os.path.join(os.path.dirname(__file__), 'data', 'result_step1.txt'), 'r')
        self.rects = {}
        for l in annoated_file.readlines():
            record = l.strip().split(' ')
            self.rects[record[0]] = [(int(record[1]), int(record[2])),
                                     (int(record[3]), int(record[4]))]
        annoated_file.close()
        annoated_file = open(os.path.join(os.path.dirname(__file__), 'data', 'result_step2.txt'), 'r')
        self.rects2 = {}
        for l in annoated_file.readlines():
            record = l.strip().split(' ')
            self.rects2[record[0]] = [(int(record[1]), int(record[2])),
                                      (int(record[3]), int(record[4]))]
        annoated_file.close()
        # gei step 3 location
        annoated_file = open(os.path.join(os.path.dirname(
            __file__), 'data', 'result_step3.txt'), 'r')
        self.rects3 = {}
        for l in annoated_file.readlines():
            record = l.strip().split(' ')
            self.rects3[record[0]] = [(int(record[1]), int(record[2])),
                                      (int(record[3]), int(record[4]))]
        annoated_file.close()
        self.mg = LineMaskGenerator()

        # self.ellipseDetector = ed.ellipse_detection()

    def cut_parts(self, img, resize, dx=0, dy=0):
        '''
        切割印刷区域内容
        :param img:待切割的图片s
        :param resize:切割图片与原始图片尺寸的比例
        :param dx:x方向偏移
        :param dy:y方向偏移
        :return:ret为切割后所有区域的图片list,pts为切割后所有区域的坐标位置list
        '''
        ret = {}
        pts = {}
        for elem in self.rects:
            rect = self.rects[elem]
            left = int((rect[0][0]) * resize) + dx
            top = int((rect[0][1]) * resize) + dy
            right = int((rect[1][0]) * resize) + dx
            bottom = int((rect[1][1]) * resize) + dy
            im = cv2.warpPerspective(img, np.float32([[1, 0, -left], [0, 1, -top], [0, 0, 1]]),
                                     (right - left, bottom - top))
            ret[elem] = im
            pts[elem] = [left, top, right, bottom]
        return ret, pts

    def cut_parts2(self, img, resize, dx=0, dy=0):
        '''
        切割打印区域内容
        :param img:待切割的图片
        :param resize:切割图片与原始图片尺寸的比例
        :param dx:x方向偏移
        :param dy:y方向偏移
        :return:ret为切割后所有区域的图片list
                 pts为切割后所有区域的坐标位置list
        '''
        ret = {}
        pts = {}
        for elem in self.rects2:
            rect = self.rects2[elem]
            left = int((rect[0][0]) * resize) + dx
            top = int((rect[0][1]) * resize) + dy
            right = int((rect[1][0]) * resize) + dx
            bottom = int((rect[1][1]) * resize) + dy
            im = cv2.warpPerspective(img, np.float32([[1, 0, -left], [0, 1, -top], [0, 0, 1]]),
                                     (right - left, bottom - top))
            ret[elem] = im
            pts[elem] = [left, top, right, bottom]
        return ret, pts

    def removeStencil(self, img, stencils, resize):
        '''
        将输入图片去除掉表格线，无关区域
        :param img:待处理的图片
        :param stencils:无关区域
        :param resize:切割图片与原始图片尺寸的比例
        :return:ret为去除无关区域的图片
                 mask为表格线蒙版
                 contentMask为无关区域蒙版
        '''
        mask, content = self.mg.genMask(img)
        raw = content.copy()
        contentMask = np.zeros((content.shape[0], content.shape[1]), np.uint8)
        for name in stencils:
            stencil = stencils[name]
            left = int((stencil[0][0]) * resize)
            top = int((stencil[0][1]) * resize)
            right = int((stencil[1][0]) * resize)
            bottom = int((stencil[1][1]) * resize)
            cv2.rectangle(content, (left, top), (right, bottom), (0, 0, 0), -1)
            cv2.rectangle(contentMask, (left, top), (right, bottom), 255, -1)
        xor = cv2.bitwise_xor(raw, content)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(xor, kernel)
        ret = cv2.bitwise_and(255 - dilated, content)

        return ret, mask, contentMask

    def match_template(self, im):
        '''
        将输入图片进行模板对齐
        :param im:待对齐的图片
        :return:img1为对齐后的图片
                 M为逆变换矩阵
                 pts为切割后所有区域的坐标位置list
                 rescale为图片放缩尺寸
                 resize为模板比例，主要用于加速优化
        '''
        rescale = 3000.0 / im.shape[1]  # 4000
        color_img = cv2.resize(im, None, None, rescale, rescale)
        self.gray_img = cv2.resize(color_img, None, None, 0.5, 0.5)
        resize = 0.5
        kp, des = self.m.sift_fet(self.gray_img)
        img1, M = self.m.post_match(self.kp1, self.des1, kp, des, color_img, input_roi=None, outputsize=(2700, 1500),
                                    offset=(0, 0), threshold=0.9, good_num=10, resize=resize)
        img1[np.where(img1 == 0)] = np.average(img1[np.where(img1 != 0)])
        return img1, M, resize, rescale

    def match_content(self, aligned, binary, tableMask, resize):
        '''
        发票中打印字段需要进行二次对齐
        :param aligned:对齐后的图片
        :param binary:对齐后的图片二值化结果
        :param tableMask:表格蒙版
        :param resize:原图与当前图片的放缩比例
        :return:img2二次对齐后的图片
                 binary2二次对齐后对应的二值化结果图
                 tableMask2为二次对齐的表格蒙版
                 np.linalg.inv(M)为逆变换
        '''
        dx, dy, cx, cy = offset(binary, self.rects['dingbiao'], 1.0 / resize)

        cos_theta = dx / np.sqrt(dx ** 2 + dy ** 2)
        sin_theta = dy / np.sqrt(dx ** 2 + dy ** 2)
        M1 = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        M2 = np.float32([[cos_theta, sin_theta, 0],
                         [-sin_theta, cos_theta, 0], [0, 0, 1]])
        M3 = np.float32([[1, 0, 571], [0, 1, 127], [0, 0, 1]])
        M = np.dot(M3, np.dot(M2, M1))
        img2 = cv2.warpPerspective(
            aligned, M, (aligned.shape[1], aligned.shape[0]))
        binary2 = cv2.warpPerspective(
            binary, M, (aligned.shape[1], aligned.shape[0]))
        tableMask2 = cv2.warpPerspective(
            tableMask, M, (aligned.shape[1], aligned.shape[0]))
        return img2, binary2, tableMask2, np.linalg.inv(M)

    def find_coarseRedRegion(self, img2, binary2):
        '''
        粗去章
        :param img2:输入图片
        :param binary2:输入图片对应的二值图
        :return:remaining_binary为去除红章的二值图
                 img2_r为红章蒙版
        '''
        img2_r = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:, :, 0]
        img2_r = ((img2_r * 1.0 / np.max(img2_r)) * 255).astype(np.uint8)
        idx = np.where((img2_r > 235))
        img2_r[...] = 0
        img2_r[idx] = 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))  # cv2.MORPH_ELLIPSE
        kernel1 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))  # cv2.MORPH_ELLIPSE
        img2_r = cv2.erode(img2_r, kernel)
        img2_r = cv2.GaussianBlur(img2_r, (7, 7), 0)
        img2_r[np.where(img2_r > 0)] = 255
        remaining_binary = cv2.bitwise_and(binary2, 255 - img2_r)
        return remaining_binary, img2_r

    def find_fineRedRegion(self, img2, binaryImg):
        '''
        细去章
        :param img2:输入图片
        :param binary2:输入图片对应的二值图
        :return:remaining_binary为去除红章的二值图
                 img2_r为红章蒙版
        '''
        
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, element)
        temp_img = img2.copy()
        # for i in range(temp_img.shape[0]):
        #     for j in range(temp_img.shape[1]):
        #         if binaryImg[i, j] == 0:
        #             temp_img[i, j] = (0, 0, 0)

        temp_img[np.where(binaryImg == 0)] = 0
        temp_img_r = cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV)[:, :, 0]
        temp_img_r = ((temp_img_r * 1.0 / np.max(temp_img_r))
                      * 255).astype(np.uint8)
        idx = np.where((temp_img_r > 225))
        temp_img_r[...] = 0
        temp_img_r[idx] = 255
        temp_img_r = cv2.GaussianBlur(temp_img_r, (5, 5), 0)
        temp_img_r[np.where(temp_img_r > 0)] = 255
        return cv2.bitwise_and(binaryImg, 255 - temp_img_r), temp_img_r

    def cut(self, im):
        '''
        字段切割
        :param im:输入图片
        :return:img_dict为切割后的图片字典,key为字段名,value为图片
                coord_dict为切割后的图片在原图中的位置字典,key为字段名,value为图片
        '''
        
        # align image with table structure
        img1, M1, resize, rescale = self.match_template(im.copy())
        M1 = np.dot(np.float32(
            [[1.0 / rescale, 0, 0], [0, 1.0 / rescale, 0], [0, 0, 1]]), M1)

        # cut Photocopying parts
        imgs1, pts1 = self.cut_parts(img1, 1.0 / resize)
        # remove table structure
        aligned = img1.copy()
        binary, mask, contentMask = self.removeStencil(
            aligned, self.rects3, 1.0 / resize)
        # align image with stationary region
        img2, binary2, mask1, M2 = self.match_content(
            aligned, binary, mask, resize)
        M2 = np.dot(M1, M2)

        # remove stamp
        withoutRedRegion, redRegion = self.find_coarseRedRegion(img2, binary2)
        withoutRedRegion2, redRegion2 = self.find_fineRedRegion(
            img2, withoutRedRegion)
        # cut print parts
        imgs2, pts2 = self.cut_parts2(img2, 1.0 / resize)
        # fine cut
        FK = Fine_cut(img1, pts1, img2, pts2,
                      withoutRedRegion2, mask1, None, M1, M2)
        img_dict, coord_dict = FK.refine()

        # 普通的章
        # img_dict1, coord_dict1 = self.ellipseDetector.cut(im)

        # for key in img_dict1:
        #     if img_dict1[key] != []:
        #         img_dict[key] = [[img_dict1[key]]]
        #         coord_dict[key] = [[coord_dict1[key]]]
        #     else:
        #         img_dict[key] = []
        #         coord_dict[key] = []
        # _DaiKaiJiGuanGaiZhang 特殊
        img_dict2, coord_dict2 = DaiKaiStampDetection.detect_daikai_stamp(im)
        for key in img_dict2:
            if img_dict2[key] != []:
                img_dict[key] = [[img_dict2[key]]]
                coord_dict[key] = [[coord_dict2[key]]]
            else:
                img_dict[key] = []
                coord_dict[key] = []

        return img_dict, coord_dict
