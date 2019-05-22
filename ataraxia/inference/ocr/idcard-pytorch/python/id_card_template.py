import cv2
import numpy as np
import base64
import time
import logging
import os

from idcard_seg import IDCardSeg
from idcard_reco import idcard_reco
from idcard_post import idcard_post
import config


class id_card(object):
    def __init__(self):
        self.id_seg = IDCardSeg()
        self.id_reco = idcard_reco()
        self.id_post = idcard_post()

    def idcard_dect(self,img):
        img1,rect1 ,rect2 = self.id_seg.idcard_seg(img)
        preds = self.id_reco.predict(img1,rect1)
        json = self.id_post.postProcessing(preds)
        return json

__HOST = '0.0.0.0'
__PORT = 7000

class IDCardRecognizeHandler(object):
    def __init__(self):
        self.predictor = id_card()

    def recognize(self, img):
        img = base64.b64decode(img);
        npimg = np.fromstring(img, dtype=np.uint8);
        cvimg = cv2.imdecode(npimg, 1)
        return self.predictor.idcard_dect(cvimg)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    handler = IDCardRecognizeHandler()
    print('Starting the rpc server at', __HOST,':', __PORT)

