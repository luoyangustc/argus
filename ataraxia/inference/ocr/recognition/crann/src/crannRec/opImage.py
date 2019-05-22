#coding:UTF-8
import numpy as np
import cv2
import math


def genShiftMatrix(src_point, dst_point):
    shift_x, shift_y = src_point - dst_point
    M = np.array([[1, 0, shift_x],
                  [0, 1, shift_y],
                  [0, 0, 1]], dtype=np.float32)
    return M

def genRotateMatrix(angle):
    '''
    angle是角度
    :param angle:
    :return:
    '''
    angle1 = angle * np.pi / 180
    a = math.cos(angle1)
    b = math.sin(angle1)

    M = np.array([[ a,-b, 0],
                  [ b, a, 0],
                  [ 0, 0, 1]], dtype=np.float32)
    return M

def genRotateAroundMatrix(center, angle):
    origin = np.array([0, 0], dtype=np.float32)
    M1 = genShiftMatrix(origin, center)
    M2 = genRotateMatrix(angle)
    M3 = genShiftMatrix(center, origin)
    M = M3.dot(M2.dot(M1))
    return M

def rotateImageByMatrix(img, M):
    h, w = img.shape[:2]
    size = (w, h)
    img1 = cv2.warpPerspective(img, M, size,flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img1

