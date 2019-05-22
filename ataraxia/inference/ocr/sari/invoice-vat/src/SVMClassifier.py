# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
#from sklearn import svm
from sklearn.externals import joblib

def imread(path, mode=-1):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)

#class predictImageCls(img, path):

class clsPredictor():
    def __init__(self, model_path):
        self.clf = joblib.load(model_path)

    def predict(self, img):
        try:
            winSize = (32, 32)
            blockSize = (16, 16)
            blockStride = (16, 16)
            cellSize = (8, 8)
            nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 64
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                    histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
            height = 640
            img0 = np.zeros((height, height, 3), dtype=np.uint8)

            if img.shape[0] > img.shape[1]:
                img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
                img0[:, int((height - img.shape[1]) / 2):int((height - img.shape[1]) / 2) + img.shape[1]] = img.copy()
            else:
                img = cv2.resize(img, (height, int(img.shape[0] * height / img.shape[1])))
                img0[int((height - img.shape[0]) / 2):int((height - img.shape[0]) / 2) + img.shape[0], :] = img.copy()

            img0[np.where(img0 == (0, 0, 0))] = np.average(img0[np.where(img0 != (0, 0, 0))])
            # cv2.imshow('img0', img0)
            # cv2.waitKey(0)

            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            # _, img0 = cv2.threshold(img0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # cv2.imshow('img0', img0)
            # cv2.waitKey(0)
            # desc = hog.compute(img0)

            winStride = (32, 32)
            padding = (8, 8)
            # locations = ((10, 20),)
            hist = hog.compute(img0, winStride, padding)  # , locations)
            hist = hist[:, 0]

            predict = self.clf.predict([hist])
        except Exception:
            predict = None

        return predict


if __name__ == '__main__':
    model_path = ''
    predictor = clsPredictor(r'D:\Users\wanghong\PycharmProjects\LearnTensorflow\FaPiaoRec\ZenZhiShuiPuTong\pt\Model\LianCi\cls.pkl')


    root_path = r'D:\Users\wanghong\PycharmProjects\LearnTensorflow\FaPiaoRec\ZenZhiShuiPuTong\pt'
    train_folder = os.path.join(root_path, 'LianCi') #DaiKai
    sub_folders = os.listdir(train_folder)

    for i, folder in enumerate(sub_folders):
        sub_folder = os.path.join(train_folder, folder)
        img_names = os.listdir(sub_folder)
        cls = folder
        for j, img_name in enumerate(img_names):
            img_path = os.path.join(sub_folder, img_name)
            img = imread(img_path)
            if predictor.predict(img) != cls:
                print('n')



# if __name__ == '__main__':
#     train_folder = r'G:\其他数据集\DaLei'
#     model_path = 'Model'
#
#     predictor = predictImageCls(os.path.join(model_path, 'cls.pkl'))
#
#     #clf = joblib.load(os.path.join(model_path, 'cls.pkl'))
#
#     sub_folders = os.listdir(train_folder)
#     for i, folder in enumerate(sub_folders):
#
#         sub_folder = os.path.join(train_folder, folder)
#         img_names = os.listdir(sub_folder)
#
#         for j, img_name in enumerate(img_names):
#             img_path = os.path.join(sub_folder, img_name)
#             img_path = r'C:\Users\wanghong\Desktop\Test_Example\qingdan\15.jpg'
#             img = imread(img_path)
#
#             #cls = getImageCls(img, clf)
#             cls = predictor.getImageCls(img)
#
#             print(cls)
#             cv2.imshow('img', img)
#             cv2.waitKey(0)