# -*- coding: utf-8 -*-
import cv2
import numpy as np

if cv2.__version__[0] == '2':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')

def show(name, img, is_debug):
    if is_debug:
        img1 = cv2.resize(img, (800, int(img.shape[0]*800/img.shape[1])))
        cv2.imshow(name, img1)
        #cv2.waitKey(0)

def imread(path, mode=-1):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)

def imwrite(path, pic):
    cv2.imencode('.png', pic)[1].tofile(path)

def findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    if cv2.__version__[0] == '2':
        contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)  # 这个函数会改变原图像，注意
    elif cv2.__version__[0] == '3':
        _, contours2, hierarchy2 = cv2.findContours(img.copy(), mode, method)  # 这个函数会改变原图像，注意
    return contours2, hierarchy2

def HoughLines(img, rho, theta, threshold):
    lines = cv2.HoughLines(img, rho, theta, threshold)
    #print('lines', lines)
    if cv2.__version__[0] == '3':
        lines = np.transpose(lines, (1, 0, 2))
        #print(lines)
    return lines
