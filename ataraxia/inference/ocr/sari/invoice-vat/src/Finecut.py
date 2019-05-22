# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import evals.src.utils as utils
from io import open
from Boxsolution import Box
import KeyList


class Fine_cut(object):
    '''

    '''
    def __init__(self, step1Img, step1RoIs, step2Img, step2RoIs, binaryImg,tableMask,fixedContentMask,M1,M2):
        '''
        构造函数
        :param step1Img:第一次对齐图，提取印刷字段
        :param step1RoIs:印刷字段位置
        :param step2Img:第二次对齐图，提取打印字段
        :param step2RoIs:打印字段位置
        :param binaryImg:第二次对齐图二值图
        :param tableMask:表格线蒙版
        :param fixedContentMask:无关字段蒙版
        :param M1:第一次变换逆矩阵
        :param M2:第二次变换逆矩阵
        '''
        self.step1Img = step1Img
        self.step1RoIs = step1RoIs
        self.step2Img = step2Img
        self.step2RoIs = step2RoIs
        self.binaryImg = binaryImg
        self.tableMask = tableMask
        self.fixedContentMask = fixedContentMask
        self.retImageDict = {}
        self.retCoordDict = {}
        self.outputGrayImg = 255 - cv2.cvtColor(step1Img, cv2.COLOR_BGR2GRAY) #- self.tableMask
        self.outputGrayImg2 = cv2.bitwise_and(255 - cv2.cvtColor(step2Img, cv2.COLOR_BGR2GRAY), 255-self.tableMask)
        self.M_1st = M1
        self.M_2nd = M2
        self.box = Box()
        self.initDicts()

    def initDicts(self):
        '''
        清空输出结果
        :return:
        '''
        self.retImageDict = {}
        self.retCoordDict = {}
        for key in KeyList.key_list:
            self.retImageDict[key] = []
            self.retCoordDict[key] = []

    def refine(self):
        '''
        细切割字段
        :return:
        '''
        self.initDicts()
        self._FaPiaoYinZhiPiHanJiYinZhiGongSi()
        self.FetchText(Field_name='_FaPiaoHaoMa_YinShua',R=70,DX=41,DY=11,Height=20,findTextRegion=True,filterSmallBoxes=True,SORT=1,singleImgList=True,singleBoxRecovery=True,Matrix=self.M_1st)
        self.FetchText(Field_name='_FaPiaoDaiMa_YinShua',DX=41,DY=11,Height=20,findTextRegion=True,filterSmallBoxes=True,SORT=1,singleImgList=True,singleBoxRecovery=True,Matrix=self.M_1st)
        self.FetchText(Field_name='_DanZhengMingCheng',DX=55,DY=11,Height=20,findTextRegion=True,filterSmallBoxes=True,SORT=1,singleImgList=True,singleBoxRecovery=True,Matrix=self.M_1st)
        self._Mima()
        self._GouMaiFang()
        self._XiaoShouFang()
        self._BeiZhuQu()
        self._ShouKuanRen ()
        self._FuHeRen()
        self._KaiPiaoRen()
        self._DaiKaiBiaoShi()
        self._JiaoYanMa()
        self._JiQiBianHao()
        self._KaiPiaoRiQi()
        self._FaPiaoHaoMa_DaYin()
        self._FaPiaoDaiMa_DaYin()
        self._JiaShuiHeJi_DaXie()
        self._JiaShuiHeJi_XiaoXie()
        self._HeJiJinE_BuHanShui()
        self._HeJiShuiE()
        self._FaPiaoJianZhiZhang()
        self._FaPiaoLianCi()
        self.Table_v2()
        return self.retImageDict,self.retCoordDict

    def getImg(self, img, pts, name, l=0, r=0, t=0, b=0,allowedZero = False,needAllRegion = False):
        '''
        截取子区域
        :param img:待截取的原图
        :param pts:待截取的位置集
        :param name:待截取的字段名
        :param l:待截取区域的左偏移
        :param r:待截取区域的右偏移
        :param t:待截取区域的上偏移
        :param b:待截取区域的下偏移
        :param allowedZero:
        :param needAllRegion:
        :return: im截取结果
                  left在原图中的左上角点横坐标
                  top在原图中的左上角点纵坐标
                  right在原图中的右下角点横坐标
                  bottom在原图中的右下角点纵坐标
        '''
        rect = pts[name]
        left = int((rect[0])) - l
        top = int((rect[1])) - t
        right = int((rect[2])) + r
        bottom = int((rect[3])) + b
        # im = img[top:bottom,left:right]
        M = np.float32([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
        im = cv2.warpPerspective(img, M, (right - left, bottom - top))
        if not allowedZero:
            im[np.where(im == 0)] = np.average(im[np.where(im != 0)])
        if needAllRegion:
            return im,left,top,right,bottom
        else:
            return im,left,top

    def findTextRegion(self, im, kx=41, ky=11, method='otsu'):
        '''
        精确定位字段
        :param im:原图
        :param kx:x方向的模糊算子
        :param ky:y方向的模糊算子
        :param method:二值化方法
        :return:boxes精确字段的位置
        '''
        gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if method == 'otsu':
            _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # 'gaussian'
            binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)

        #cv2.imshow('mask', binary_img)
        mask = cv2.GaussianBlur(255 - binary_img, (kx, ky), 0)

        contours, hierarchy = utils.findContours(mask)

        boxes = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)

        for i,box in enumerate(boxes):
            d1 = np.sqrt((box[0, 0] - box[1, 0]) ** 2 + (box[0, 1] - box[1, 1]) ** 2)
            d2 = np.sqrt((box[1, 0] - box[2, 0]) ** 2 + (box[1, 1] - box[2, 1]) ** 2)
            ratio  = max(d1,d2)/min(d1,d2)
            if ratio<2:
                left = int(np.min(box[:, 0]))
                top = int(np.min(box[:, 1]))
                right = int(np.max(box[:, 0]))
                bottom = int(np.max(box[:, 1]))
                boxes[i] = np.array([[left,bottom],[left,top],[right,top],[right,bottom]])

        return boxes

    def findMultipleLineTextRegion(self, binaryImg,kx = 27,ky = 5):
        '''
        精确定位多行字段，一般用于备注区，表格区，密码区，购买方，销售方
        :param binaryImg:二值图
        :param kx:x方向的模糊算子
        :param ky:y方向的模糊算子
        :return:boxes精确字段的位置:
        '''
        mask1 = cv2.GaussianBlur(binaryImg, (kx, ky), 0)
        _contours, _hierarchy = utils.findContours(mask1)
        boxes = []
        for cnt in _contours:
            rect = cv2.minAreaRect(cnt)
            # box = cv2.cv.BoxPoints(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)

        for i,box in enumerate(boxes):
            d1 = np.sqrt((box[0, 0] - box[1, 0]) ** 2 + (box[0, 1] - box[1, 1]) ** 2)
            d2 = np.sqrt((box[1, 0] - box[2, 0]) ** 2 + (box[1, 1] - box[2, 1]) ** 2)
            ratio  = max(d1,d2)/(0.0001+ min(d1,d2))
            if ratio<2:
                left = int(np.min(box[:, 0]))
                top = int(np.min(box[:, 1]))
                right = int(np.max(box[:, 0]))
                bottom = int(np.max(box[:, 1]))
                w = right - left
                h = bottom - top
                left+= w*0.15
                right-=w*0.15
                top+=h*0.15
                bottom-=h*0.15
                boxes[i] = np.array([[left,bottom],[left,top],[right,top],[right,bottom]])
        return boxes

    def cutRotatedBox(self, im, box):
        '''
        斜区域截取
        :param im:待截取图片
        :param box:待截取区域
        :return:im截取结果
        '''
        im = im.copy()
        dx1, dy1 = (box[2, 0] - box[1, 0], box[2, 1] - box[1, 1])
        dx2, dy2 = (box[1, 0] - box[0, 0], box[1, 1] - box[0, 1])
        cx = int((box[2, 0] + box[1, 0] + box[0, 0] + box[3, 0]) / 4)
        cy = int((box[2, 1] + box[1, 1] + box[0, 1] + box[3, 1]) / 4)
        dx = 0
        dy = 0
        _dx = 0
        _dy = 0
        if abs(dx2) > abs(dx1):
            dx = dx2
            dy = dy2
            _dx = dx1
            _dy = dy1
        else:
            dx = dx1
            dy = dy1
            _dx = dx2
            _dy = dy2
        if dx < 0:
            dx = -dx
            dy = -dy
        if _dy < 0:
            _dx = -_dx
            _dy = -_dy

        w = np.sqrt(dx ** 2 + dy ** 2)
        h = np.sqrt(_dx ** 2 + _dy ** 2)

        _cos = dx / w
        _sin = dy / w
        # print _cos,_sin

        M1 = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        M2 = np.float32([[_cos, _sin, 0], [-_sin, _cos, 0], [0, 0, 1]])
        M3 = np.float32([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]])

        M = np.dot(M3, np.dot(M2, M1))
        im = cv2.warpPerspective(im, M, (int(w), int(h)))

        return im

    def singleImgList(self, im, boxes):
        '''
        截取一族区域
        :param im:原始图片
        :param boxes:一维list，其中存放了待截取的区域位置
        :return:imgs一维list，其中存放截取后的区域图片
        '''
        imgs = []
        for box in boxes:
            im1= self.cutRotatedBox(im, box)
            imgs.append(im1)
        return imgs

    def multipleImgList(self,im,_2dBoxes):
        '''
        截取一族图片
        :param im: 原始图片
        :param _2dBoxes: 二维list，其中存放了待截取的区域位置
        :return:imgs二维list，其中存放截取后的区域图片
        '''
        imgs = []
        for _1dBoxes in _2dBoxes:
            _imgs = []
            for box in _1dBoxes:
                #print box
                img1 = self.cutRotatedBox(im, box)
                _imgs.append(img1)
            if len(_1dBoxes)>0:
                imgs.append(_imgs)
        return imgs

    def _FaPiaoYinZhiPiHanJiYinZhiGongSi(self):
        '''
        截取发票印制公司
        :return:
        '''
        im,left,top = self.getImg(self.step1Img, self.step1RoIs, '_FaPiaoYinZhiPiHanJiYinZhiGongSi', l=10, r=10)
        retIm,left,top = self.getImg(self.outputGrayImg, self.step1RoIs, '_FaPiaoYinZhiPiHanJiYinZhiGongSi', l=10, r=10)
        h, w, c = im.shape
        M1 = np.float32([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        M2 = np.float32([[1, 0, 0], [0, 1, w], [0, 0, 1]])
        M = np.dot(M2, M1)
        M3  = np.linalg.inv(M)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        im = cv2.warpPerspective(im, M, (h, w))
        retIm = cv2.warpPerspective(retIm, M, (h, w))
        boxes = self.findTextRegion(im, 55, 3, method='gaussian')
        boxes = self.box.filterSmallBoxes(boxes, 20)

        boxes, _2dBoxes = self.box.sort2(boxes, 40)
        _2dBoxes = self.box.mergeBox(_2dBoxes)

        imgs = self.multipleImgList(retIm, _2dBoxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])

        new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_1st, np.dot(M4, M3)))  #
        if len(imgs) > 0:
            self.retImageDict['_FaPiaoYinZhiPiHanJiYinZhiGongSi'] = imgs
            self.retCoordDict['_FaPiaoYinZhiPiHanJiYinZhiGongSi'] = new_boxes
        else:
            self.retImageDict['_FaPiaoYinZhiPiHanJiYinZhiGongSi'] = []
            self.retCoordDict['_FaPiaoYinZhiPiHanJiYinZhiGongSi'] = []

    def _Mima(self):
        '''
        截取密码区
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, 'mimaqu',allowedZero = True,t = 20)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, 'mimaqu',allowedZero= True,t = 20)
        boxes = self.findMultipleLineTextRegion(im,kx = 41,ky = 5)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.splitBox(boxes,hSplit=50)
        boxes,_2dBoxes = self.box.sort2(boxes,40)
        _2dBoxes = self.box.mergeBox(_2dBoxes)

        imgs = self.multipleImgList(retIm, _2dBoxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_2nd, M4))

        if len(imgs)>0:
            self.retImageDict['_MiMa'] = imgs
            self.retCoordDict['_MiMa'] = new_boxes
        else:
            self.retImageDict['_MiMa'] = []
            self.retCoordDict['_MiMa'] = []

    def _GouMaiFang(self):
        '''
        截取购买方
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, 'goumaiqu', allowedZero=True,t = 20)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, 'goumaiqu', allowedZero=True,t = 20)
        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=5)

        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.splitBox(boxes, hSplit=60)

        boxes,_2dBoxes = self.box.sort2(boxes, 40)

        imgs = self.multipleImgList(retIm, _2dBoxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_2nd, M4))
        key_list = ['_GouMaiFangMingCheng', '_GouMaiFangNaShuiShiBieHao', '_GouMaiFangDiZhiJiDianHua', '_GouMaiFangKaiHuHangJiZhangHao']

        for row, row_imgs in enumerate(imgs):
            if row < 4:
                key = key_list[row]
                self.retImageDict[key] = [row_imgs]
                self.retCoordDict[key] = [new_boxes[row]]
            else:
                break

    def _XiaoShouFang(self):
        '''
        截取销售方区
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, 'xiaoshoufangqu', allowedZero=True,t = 15)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, 'xiaoshoufangqu', allowedZero=True,t = 15)
        boxes = self.findMultipleLineTextRegion(im, kx=31, ky=5)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.splitBox(boxes, hSplit=60)
        boxes, _2dBoxes = self.box.sort2(boxes, 40)
        imgs = self.multipleImgList(retIm, _2dBoxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_2nd, M4))

        key_list = ['_XiaoShouFangMingCheng', '_XiaoShouFangNaShuiRenShiBieHao', '_XiaoShouFangDiZhiJiDianHua', '_XiaoShouFangKaiHuHangJiZhangHao']

        for row, row_imgs in enumerate(imgs):
            if row < 4:
                key = key_list[row]
                self.retImageDict[key] = [row_imgs]
                self.retCoordDict[key] = [new_boxes[row]]
            else:
                break

    def _BeiZhuQu(self):
        '''
        截取备注区
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, 'beizhuqu', allowedZero=True)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, 'beizhuqu', allowedZero=True)
        boxes = self.findMultipleLineTextRegion(im, kx=55, ky=5)

        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes, _2dBoxes = self.box.sort2(boxes, 40)
        _2dBoxes = self.box.mergeBox(_2dBoxes)

        imgs = self.multipleImgList(retIm, _2dBoxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_2nd, M4))

        if len(imgs) > 0:
            self.retImageDict['_BeiZhu'] = imgs
            self.retCoordDict['_BeiZhu'] = new_boxes
        else:
            self.retImageDict['_BeiZhu'] = []
            self.retCoordDict['_BeiZhu'] = []
    
    def _ShouKuanRen(self):
        '''
        截取收款人
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_ShouKuanRen', allowedZero=True,l = 20)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_ShouKuanRen', allowedZero=True,l=20)
        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)
        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))

        if len(imgs) > 0:
            self.retImageDict['_ShouKuanRen'] = [imgs]
            self.retCoordDict['_ShouKuanRen'] = [new_boxes]
        else:
            self.retImageDict['_ShouKuanRen'] = []
            self.retCoordDict['_ShouKuanRen'] = []

    def _FuHeRen(self):
        '''
        截取复核人
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_FuHeRen', allowedZero=True, l=20)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_FuHeRen', allowedZero=True, l=20)
        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)
        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))
        if len(imgs) > 0:
            self.retImageDict['_FuHeRen'] = [imgs]
            self.retCoordDict['_FuHeRen'] = [new_boxes]
        else:
            self.retImageDict['_FuHeRen'] = []
            self.retCoordDict['_FuHeRen'] = []

    def _KaiPiaoRen(self):
        '''
        截取开票人
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_KaiPiaoRen', allowedZero=True, l=20,r=20)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_KaiPiaoRen', allowedZero=True, l=20,r=20)
        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)
        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))

        if len(imgs) > 0:
            self.retImageDict['_KaiPiaoRen'] = [imgs]
            self.retCoordDict['_KaiPiaoRen'] = [new_boxes]
        else:
            self.retImageDict['_KaiPiaoRen'] = []
            self.retCoordDict['_KaiPiaoRen'] = []

    def _HeJiShuiE(self):
        '''
        截取合计税额
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_HeJiShuiE', allowedZero=True, r=20,t=10)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_HeJiShuiE', allowedZero=True,r=20,t=10)

        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)

        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))

        if len(imgs) > 0:
            self.retImageDict['_HeJiShuiE'] = [imgs]
            self.retCoordDict['_HeJiShuiE'] = [new_boxes]
        else:
            self.retImageDict['_HeJiShuiE'] = []
            self.retCoordDict['_HeJiShuiE'] = []
    
    def test(self,dictItem):
        '''
        测试截取区域正确与否
        :param dictItem:感兴趣的区域图片集合
        :return:
        '''
        for _imgs in dictItem:
            for img in _imgs:
                cv2.imshow('win',img)

    def FetchText(self,Field_name,Matrix,R=0,DX=0,DY=0,SORT=0,Height=0,findTextRegion=False,filterSmallBoxes=False,singleImgList=False,singleBoxRecovery=False):
        '''

        :param Field_name:字段名称
        :param Matrix:变换逆矩阵
        :param R:右偏移
        :param DX:横向偏移
        :param DY:纵向偏移
        :param SORT:排序算法
        :param Height:筛选高度
        :param findTextRegion:是否调用findTextRegion
        :param filterSmallBoxes:是否调用findTextRegion
        :param singleImgList:是否调用singleImgList
        :param singleBoxRecovery:是否需要进行逆变换
        :return:
        '''
        im,left,top = self.getImg(self.step1Img, self.step1RoIs, Field_name, r=R)
        retIm,left,top = self.getImg(self.outputGrayImg, self.step1RoIs, Field_name, r=R)
        if(findTextRegion):
            boxes = self.findTextRegion(im, DX, DY)
        if(filterSmallBoxes):
            boxes = self.box.filterSmallBoxes(boxes, Height)
        if(SORT==1):
            boxes = self.box.sort(boxes)
        elif(SORT==2):
            boxes = self.box.sort2(boxes)
        if(singleImgList):
            imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        if(singleBoxRecovery):
            new_boxes = self.box.singleBoxRecovery(boxes, np.dot(Matrix,M4))
        if len(imgs)>0:
            self.retImageDict[Field_name] = [imgs]
            self.retCoordDict[Field_name] = [new_boxes]
        else:
            self.retImageDict[Field_name] = []
            self.retCoordDict[Field_name] = []


    def _DaiKaiBiaoShi(self):
        '''
        截取代开标识
        :return:
        '''
        im,left,top,right,bottom = self.getImg(self.step2Img, self.step2RoIs, '_DaiKaiBiaoShi', allowedZero=True, l=20,needAllRegion=True)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        boxes = np.float32([[[0,bottom-top],[0,0],[right-left,0],[right-left,bottom-top]]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_1st, M4))
        self.retImageDict['_DaiKaiBiaoShi'] = [[im]]
        self.retCoordDict['_DaiKaiBiaoShi'] = [new_boxes]

    def _FaPiaoJianZhiZhang(self):
        '''
        截取发票监制章
        :return:
        '''
        im,left,top,right,bottom = self.getImg(self.step1Img, self.step1RoIs, '_FaPiaoJianZhiZhang', allowedZero=True, l=20,needAllRegion=True)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        boxes = np.float32([[[0, bottom - top], [0, 0], [right - left, 0], [right - left, bottom - top]]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_1st, M4))
        self.retImageDict['_FaPiaoJianZhiZhang'] = [[im]]
        self.retCoordDict['_FaPiaoJianZhiZhang'] = [new_boxes]
    
    def _FaPiaoLianCi(self):
        im,left,top,right,bottom = self.getImg(self.step1Img, self.step1RoIs, '_FaPiaoLianCi', allowedZero=True, l=20,needAllRegion=True)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        boxes = np.float32([[[0, bottom - top], [0, 0], [right - left, 0], [right - left, bottom - top]]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_1st, M4))

        self.retImageDict['_FaPiaoLianCi'] = [[im]]
        self.retCoordDict['_FaPiaoLianCi'] = [new_boxes]

    def _JiQiBianHao(self):
        '''
        截取机器编号
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_JiQiBianHao', allowedZero=True, l=20)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_JiQiBianHao', allowedZero=True, l=20)

        boxes = self.findMultipleLineTextRegion(im, kx=13, ky=5)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)

        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))

        if len(imgs) > 0:
            self.retImageDict['_JiQiBianHao'] = [imgs]
            self.retCoordDict['_JiQiBianHao'] = [new_boxes]
        else:
            self.retImageDict['_JiQiBianHao'] = []
            self.retCoordDict['_JiQiBianHao'] = []

    def _JiaoYanMa(self):
        '''
        截取校验码
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_JiaoYanMa', allowedZero=True, l=20)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_JiaoYanMa', allowedZero=True, l=20)

        boxes = self.findMultipleLineTextRegion(im, kx=13, ky=5)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes, _2dBoxes = self.box.sort2(boxes, 40)

        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))
        if len(imgs) > 0:
            self.retImageDict['_JiaoYanMa'] = [imgs]
            self.retCoordDict['_JiaoYanMa'] = [new_boxes]
        else:
            self.retImageDict['_JiaoYanMa'] = []
            self.retCoordDict['_JiaoYanMa'] = []

        _2dBoxes_he = self.box.mergeBox(_2dBoxes)  # TODELETE
        imgs = self.multipleImgList(retIm, _2dBoxes_he)
        if len(imgs) > 0:
            self.retImageDict['_JiaoYanMaHe'] = imgs
            self.retCoordDict['_JiaoYanMaHe'] = [new_boxes]
        else:
            self.retImageDict['_JiaoYanMaHe'] = []
            self.retCoordDict['_JiaoYanMaHe'] = []

    def _KaiPiaoRiQi(self):
        '''
        截取开票日期
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_KaiPiaoRiQi', allowedZero=True,l=10,r=10,t=10)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_KaiPiaoRiQi', allowedZero=True,l=10,r=10, t = 10)

        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)

        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))
        if len(imgs) > 0:
            self.retImageDict['_KaiPiaoRiQi'] = [imgs]
            self.retCoordDict['_KaiPiaoRiQi'] = [new_boxes]
        else:
            self.retImageDict['_KaiPiaoRiQi'] = []
            self.retCoordDict['_KaiPiaoRiQi'] = []

    def _FaPiaoHaoMa_DaYin(self):
        '''
        截取发票号码打印
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_FaPiaoHaoMa_DaYin', allowedZero=True,l=10,r=10,t =5)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_FaPiaoHaoMa_DaYin', allowedZero=True,l=10,r=10,t=5)


        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)

        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))
        if len(imgs) > 0:
            self.retImageDict['_FaPiaoHaoMa_DaYin'] = [imgs]
            self.retCoordDict['_FaPiaoHaoMa_DaYin'] = [new_boxes]
        else:
            self.retImageDict['_FaPiaoHaoMa_DaYin'] = []
            self.retCoordDict['_FaPiaoHaoMa_DaYin'] = []

    def _FaPiaoDaiMa_DaYin(self):
        '''
        截取发票代码打印
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_FaPiaoDaiMa_DaYin', allowedZero=True,l=5,r=10)
        retIm,left,top= self.getImg(self.outputGrayImg2, self.step2RoIs, '_FaPiaoDaiMa_DaYin', allowedZero=True,l=5,r=10)

        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)

        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))
        if len(imgs) > 0:
            self.retImageDict['_FaPiaoDaiMa_DaYin'] = [imgs]
            self.retCoordDict['_FaPiaoDaiMa_DaYin'] = [new_boxes]
        else:
            self.retImageDict['_FaPiaoDaiMa_DaYin'] = []
            self.retCoordDict['_FaPiaoDaiMa_DaYin'] = []

    def _JiaShuiHeJi_XiaoXie(self):
        '''
        截取价税合计小写
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_JiaShuiHeJi_XiaoXie', allowedZero=True, t=15,b=15,r = 200)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_JiaShuiHeJi_XiaoXie', allowedZero=True,t=15, b=15,r = 200)

        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)

        boxes, _2dBoxes = self.box.sort2(boxes, 40)
        _2dBoxes = self.box.mergeBox(_2dBoxes)

        imgs = self.multipleImgList(retIm, _2dBoxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_2nd, M4))

        if len(imgs) > 0:
            self.retImageDict['_JiaShuiHeJi_XiaoXie'] = imgs
            self.retCoordDict['_JiaShuiHeJi_XiaoXie'] = new_boxes
        else:
            self.retImageDict['_JiaShuiHeJi_XiaoXie'] = []
            self.retCoordDict['_JiaShuiHeJi_XiaoXie'] = []

    def _JiaShuiHeJi_DaXie(self):
        '''
        截取价税合计大写
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_JiaShuiHeJi_DaXie', allowedZero=True, t=10, b=5,r = 200)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_JiaShuiHeJi_DaXie', allowedZero=True, t=10, b=5,r = 200)
        boxes = self.findMultipleLineTextRegion(im, kx=41, ky=17)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))
        if len(imgs) > 0:
            self.retImageDict['_JiaShuiHeJi_DaXie'] = [imgs]
            self.retCoordDict['_JiaShuiHeJi_DaXie'] = [new_boxes]
        else:
            self.retImageDict['_JiaShuiHeJi_DaXie'] = []
            self.retCoordDict['_JiaShuiHeJi_DaXie'] = []

    def SingleCol(self,rowNum = 8,col = 1,kx = 27,ky = 11, isSingleBox = False):
        '''
        表格区域单列截取（过期）
        :param rowNum: 行内单元数
        :param col: 当前列号
        :param kx: x方向偏移
        :param ky: y方向偏移
        :param isSingleBox:是否每个单元格内是单个感兴趣区域
        :return:
        '''
        for i in range(rowNum):
            name = 'Table'+str(i+1)+str(col)
            im,left,top = self.getImg(self.binaryImg, self.step2RoIs, name, allowedZero=True)

            retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, name, allowedZero=True)

            boxes = self.findMultipleLineTextRegion(im, kx=kx, ky=ky)
            if col != 4:
                boxes = self.box.filterSmallBoxes(boxes, 25)
            if isSingleBox:
                boxes = self.box.filterSmallBoxes(boxes, 25)
                boxes = self.box.getMaxRegionBox(boxes)

                imgs = self.singleImgList(retIm, boxes)
                M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
                new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))

                if len(imgs) > 0:
                    self.retImageDict[name] = [imgs]
                    self.retCoordDict[name] = [new_boxes]
                else:
                    self.retImageDict[name] = []
                    self.retCoordDict[name] = []
            else:
                boxes, _2dBoxes = self.box.sort2(boxes, 40)
                _2dBoxes = self.box.mergeBox(_2dBoxes)

                imgs = self.multipleImgList(retIm, _2dBoxes)
                M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
                new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_2nd, M4))

                if len(imgs) > 0:
                    self.retImageDict[name] = imgs
                    self.retCoordDict[name] = new_boxes
                else:
                    self.retImageDict[name] = []
                    self.retCoordDict[name] = []

    def _HeJiJinE_BuHanShui(self):
        '''
        截取合计金额不含税
        :return:
        '''
        im,left,top = self.getImg(self.binaryImg, self.step2RoIs, '_HeJiJinE_BuHanShui', allowedZero=True, r=20,t=10)
        retIm,left,top = self.getImg(self.outputGrayImg2, self.step2RoIs, '_HeJiJinE_BuHanShui', allowedZero=True, r=20,t=10)

        boxes = self.findMultipleLineTextRegion(im, kx=27, ky=11)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.getMaxRegionBox(boxes)

        imgs = self.singleImgList(retIm, boxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.singleBoxRecovery(boxes, np.dot(self.M_2nd, M4))
        if len(imgs) > 0:
            self.retImageDict['_HeJiJinE_BuHanShui'] = [imgs]
            self.retCoordDict['_HeJiJinE_BuHanShui'] = [new_boxes]
        else:
            self.retImageDict['_HeJiJinE_BuHanShui'] = []
            self.retCoordDict['_HeJiJinE_BuHanShui'] = []

    def Table(self):
        '''
        表格截取（过期）
        :return:
        '''
        self.SingleCol(col=1, isSingleBox=True)
        self.SingleCol(col=2, isSingleBox=False)
        self.SingleCol(col=3, isSingleBox=True)
        self.SingleCol(col=4, isSingleBox=True)
        self.SingleCol(col=5, isSingleBox=True)
        self.SingleCol(col=6, isSingleBox=True)
        self.SingleCol(col=7, isSingleBox=True)
        self.SingleCol(col=8, isSingleBox=True)

    def Table_v2(self):
        '''
        表格截取
        :return:
        '''
        self.SingleCol_v2(col=1)
        self.SingleCol_v2(col=2)
        self.SingleCol_v2(col=3,is_Max=True)
        self.SingleCol_v2(col=4,is_Max=True)
        self.SingleCol_v2(col=5,is_Max=True)
        self.SingleCol_v2(col=6,is_Max=True)
        self.SingleCol_v2(col=7,is_Max=True)
        self.SingleCol_v2(col=8,is_Max=True)

    def SingleCol_v2(self,col = 1,kx = 35,ky = 9,is_Max = False):
        '''
        表格区域单列截取
        :param col:所在列号
        :param kx: x方向偏移
        :param ky: y方向偏移
        :param is_Max:单元格内是否只截取最大区域
        :return:
        '''
        name = 'Tablev2_%d'%(col)
        im, left, top = self.getImg(self.binaryImg, self.step2RoIs, name, allowedZero=True)
        retIm, left, top = self.getImg(self.outputGrayImg2, self.step2RoIs, name, allowedZero=True)

        boxes = self.findMultipleLineTextRegion(im, kx=kx, ky=ky)
        boxes = self.box.filterSmallBoxes(boxes, 20)
        boxes = self.box.splitBox(boxes, hSplit=50)
        boxes, _2dBoxes = self.box.sort2(boxes, 40)

        if is_Max:
            _2dBoxes = self.box.get2DMaxRegionBox(_2dBoxes)
        else:
            _2dBoxes = self.box.mergeBox(_2dBoxes)

        imgs = self.multipleImgList(retIm, _2dBoxes)
        M4 = np.float32([[1, 0, left], [0, 1, top], [0, 0, 1]])
        new_boxes = self.box.multipleBoxRecovery(_2dBoxes, np.dot(self.M_2nd, M4))

        if len(imgs) > 0:
            self.retImageDict[name] = imgs
            self.retCoordDict[name] = new_boxes
        else:
            self.retImageDict[name] = []
            self.retCoordDict[name] = []