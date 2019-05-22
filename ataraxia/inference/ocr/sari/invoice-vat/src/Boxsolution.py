# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import evals.src.utils
from io import open
from functools import cmp_to_key

class Box(object):
    '''
    功能：用于处理所有旋转区域
    备注：每个旋转区域由四个点组成[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    '''
    def filterSmallBoxes(self, boxes, minHeight=-1):  # Height,Width
        '''
        当每个box的高度小于minHeight时，将从boxes中删除
        :param boxes:存有一族box的list
        :param minHeight:box最小高度
        :return:ret返回一族过滤后的box的list
        '''
        ret = []
        for box in boxes:
            d1 = np.sqrt((box[0, 0] - box[1, 0]) ** 2 +
                         (box[0, 1] - box[1, 1]) ** 2)
            d2 = np.sqrt((box[1, 0] - box[2, 0]) ** 2 +
                         (box[1, 1] - box[2, 1]) ** 2)
            h = 0
            w = 0
            if d1 < d2:
                h = d1
                w = d2
            else:
                h = d2
                w = d1
            if minHeight != -1:
                if h < minHeight:
                    continue
            ret.append(box)
        return ret

    def getMaxRegionBox(self, boxes):
        '''
        在boxes中选择一个面积最大的box将其返回
        :param boxes:一族box的list
        :return:ret返回一个box
        '''
        ret = []
        maxArea = 0
        idx = -1
        for i, box in enumerate(boxes):
            d1 = np.sqrt((box[0, 0] - box[1, 0]) ** 2 +
                         (box[0, 1] - box[1, 1]) ** 2)
            d2 = np.sqrt((box[1, 0] - box[2, 0]) ** 2 +
                         (box[1, 1] - box[2, 1]) ** 2)
            h = 0
            w = 0
            if d1 < d2:
                h = d1
                w = d2
            else:
                h = d2
                w = d1
            area = h * w
            if area > maxArea:
                maxArea = area
                idx = i
        if idx != -1:
            ret.append(boxes[idx])
        return ret

    def get2DMaxRegionBox(self, _2dboxes):
        '''
        在二维list中，选择出每行中面积最大的box，将它们返回
        :param _2dboxes:_2dboxes存放box的二维list
        :return:_2d_ret存放box的一维list
        '''
        _2d_ret = []
        for _1dboxes in _2dboxes:
            _1d_res = self.getMaxRegionBox(_1dboxes)
            _2d_ret.append(_1d_res)
        return _2d_ret

    def sort(self, boxes):
        '''
        将boxes中的box按照从左到右，从上到下进行排序
        :param boxes:存有一族box的list
        :return:ret返回一族排序后的box的list
        '''
        ret = []
        centers = []
        for box in boxes:
            cx = int((box[2, 0] + box[1, 0] + box[0, 0] + box[3, 0]) / 4)
            cy = int((box[2, 1] + box[1, 1] + box[0, 1] + box[3, 1]) / 4)
            centers.append(cy * 1000 + cx)
        order = np.argsort(centers)
        ret = []
        for i in order:
            ret.append(boxes[i])
        return ret

    def sort2(self, boxes, boxHeight):
        '''
        将boxes按照从左到右，从上到下进行排序，当两个box的y方向的偏差小于boxHeight一半时认为他们属于一行，
          将排序后的结果组织成二维list retlist中
        :param boxes:排序后的一维box的list,ret_list按行列排序后的二维list
        :param boxHeight:为行高
        :return:boxes排序后的一维box的list,ret_list按行列排序后的二维list
        '''
        if len(boxes) == 0:
            return [], []
        def cmp_items(a, b):
            ax = int((a[2, 0] + a[1, 0] + a[0, 0] + a[3, 0]) / 4)
            ay = int((a[2, 1] + a[1, 1] + a[0, 1] + a[3, 1]) / 4)
            bx = int((b[2, 0] + b[1, 0] + b[0, 0] + b[3, 0]) / 4)
            by = int((b[2, 1] + b[1, 1] + b[0, 1] + b[3, 1]) / 4)
            if by > ay + boxHeight / 2:
                return 1
            elif by < ay - boxHeight / 2:
                return -1
            else:
                if bx > ax:
                    return 1
                elif bx < ax:
                    return -1
                else:
                    return 0
        boxes.sort(key=cmp_to_key(cmp_items), reverse=True)
        if len(boxes) == 0:
            return []

        ret_list = []
        cur_list = []
        ret_list.append(cur_list)
        for i in range(len(boxes) - 1):
            a = boxes[i]
            b = boxes[i + 1]
            ay = int((a[2, 1] + a[1, 1] + a[0, 1] + a[3, 1]) / 4)
            by = int((b[2, 1] + b[1, 1] + b[0, 1] + b[3, 1]) / 4)
            if by > ay + boxHeight / 2:
                cur_list.append(a)
                cur_list = []
                ret_list.append(cur_list)
            else:
                cur_list.append(a)
        cur_list.append(boxes[-1])

        return boxes, ret_list

    def mergeBox(self, input_2dBoxes):
        '''
        将input_2dBoxes中同一行中的box
        :param input_2dBoxes:二维box的list
        :return:一维box的list
        '''
        output_2dBoxes = []
        for _1dBoxes in input_2dBoxes:
            if len(_1dBoxes) > 0:
                l = _1dBoxes[0]
                r = _1dBoxes[-1]

                clx = np.mean(l[:, 0])
                cly = np.mean(l[:, 1])
                crx = np.mean(r[:, 0])
                cry = np.mean(r[:, 1])

                try:
                    p1 = l[np.where((l[:, 0] < clx) & (l[:, 1] > cly))][0]
                    p2 = l[np.where((l[:, 0] < clx) & (l[:, 1] < cly))][0]
                    p3 = r[np.where((r[:, 0] > crx) & (r[:, 1] < cry))][0]
                    p4 = r[np.where((r[:, 0] > crx) & (r[:, 1] > cry))][0]
                    output_2dBoxes.append([np.array([p1, p2, p3, p4])])
                except:
                    output_2dBoxes.append(_1dBoxes)

        return output_2dBoxes

    def splitBox(self, boxes, hSplit=50):
        '''
        当boxes中box高度大于hSplit时，就要进行分行，使得每行高度小于hSplit,如果不进行分行直接加入到ret_boxes中，如果分行则将分割后的结果添加到ret_boxes中
        :param boxes:一族box的list
        :param hSplit:行高
        :return:存有一族box的list
        '''
        ret_boxes = []
        for box in boxes:
            d1 = np.sqrt((box[0, 0] - box[1, 0]) ** 2 +
                         (box[0, 1] - box[1, 1]) ** 2)
            d2 = np.sqrt((box[1, 0] - box[2, 0]) ** 2 +
                         (box[1, 1] - box[2, 1]) ** 2)
            h = 0
            w = 0
            if d1 < d2:
                h = d1
                w = d2
                source_box = box
            else:
                h = d2
                w = d1
                source_box = np.vstack((box[1:, :], box[0:1, :]))

            split_num = int(np.ceil(h * 1.0 / hSplit))
            p1 = source_box[0, :]
            p2 = source_box[1, :]
            p3 = source_box[2, :]
            p4 = source_box[3, :]

            for i in range(split_num):
                ratio = i * 1.0 / split_num
                ratio2 = (i + 1) * 1.0 / split_num

                _p1 = p1 * (1 - ratio) + p2 * ratio
                _p2 = p1 * (1 - ratio2) + p2 * ratio2
                _p3 = p3 * ratio2 + p4 * (1 - ratio2)
                _p4 = p3 * ratio + p4 * (1 - ratio)
                _box = np.float32([_p1, _p2, _p3, _p4])
                ret_boxes.append(_box)

        return ret_boxes

    def singleBoxRecovery(self, boxes, M):
        '''
        将box进行变换，计算在新坐标系下的位置，一般用于将现有坐标恢复到原图中
        :param boxes:存有一族box的list
        :param M:坐标变换矩阵
        :return:ret_boxes返回一族变换后的box的list
        '''
        ret_boxes = []
        for box in boxes:
            coord_old = np.vstack((box.T, np.ones((1, 4))))
            coord_new = np.dot(M, coord_old)
            coord_new[0, :] /= coord_new[2, :]
            coord_new[1, :] /= coord_new[2, :]
            coord_new = coord_new[0:2, :].T
            ret_boxes.append(coord_new)
        return ret_boxes

    def multipleBoxRecovery(self, _2dBoxes, M):
        '''
        将box进行变换，计算在新坐标系下的位置，一般用于将现有坐标恢复到原图中
        :param _2dBoxes:一族box的二维list
        :param M:坐标变换矩阵
        :return:将box进行变换，计算在新坐标系下的位置，一般用于将现有坐标恢复到原图中
        '''
        ret_2dBoxs = []
        for _1dBoxes in _2dBoxes:
            _boxes = []
            for box in _1dBoxes:
                coord_old = np.vstack((box.T, np.ones((1, 4))))
                coord_new = np.dot(M, coord_old)
                coord_new[0, :] /= coord_new[2, :]
                coord_new[1, :] /= coord_new[2, :]
                coord_new = coord_new[0:2, :].T
                _boxes.append(coord_new)
            if len(_1dBoxes) > 0:
                ret_2dBoxs.append(_boxes)
        return ret_2dBoxs
