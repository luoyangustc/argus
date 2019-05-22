# -*- coding: utf-8 -*-
import cv2
import numpy as np
import base64
import time
import logging
import os
import json

from idcard_seg import IDCardSeg
from idcard_reco import idcard_reco
from idcard_post import idcard_post
from config import config
from digits_detect import digitDetect
import sys


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


from crnnport import crnnSource, crnnRec_single
class ocr_reco(object):
	def __init__(self,model_path):
		self.model, self.converter = crnnSource(model_path)
		self.digits_model = digitDetect(config.RECOGNITION.DIGITS_MODEL_PATH,
										config.RECOGNITION.DIGITS_MIN_BIT,
										config.RECOGNITION.DIGITS_MAX_BIT)

	def predict(self, img, pt,):
		predicts = []

		left = pt[0][0]
		top = pt[0][1]
		right = pt[2][0]
		bottom = pt[2][1]
		field_img = img[top:bottom,left:right]
#		cv2.imwrite("test.jpg",field_img)
		field_predict = crnnRec_single(self.model, self.converter, field_img)


		return field_predict

	def predict_digits(self,img,pt):
		left = pt[0][0]
		top = pt[0][1]
		right = pt[2][0]
		bottom = pt[2][1]
		field_img = img[top:bottom, left:right]
		#		cv2.imwrite("test.jpg",field_img)
		field_predict, probs = self.digits_model.digits_predict(field_img)

		return field_predict



class template_matcher(object):
    def __init__(self):
        self.uuid = -1
        from idcard_seg import IDCardMatcher
        self.Match = IDCardMatcher()
        self.points =[]
        self.descripts = []
        self.recog = ocr_reco(config.RECOGNITION.ADDRESS_MODEL_PATH)

    def init_matcher(self,template_dict):
        rect = template_dict['Rects']
        if "," in template_dict['img']:
            img = base64.b64decode(template_dict['img'].split(",")[1])
        else:
            img = base64.b64decode(template_dict['img'])
        npimg = np.fromstring(img, dtype=np.uint8)
        cvimg = cv2.imdecode(npimg, 1)

#        pt =template_dict["fields"][0]['loc']

#        left = pt[0][0]
#        top = pt[0][1]
#        right = pt[2][0]
#        bottom = pt[2][1]
#        field_img = cvimg[top:bottom, left:right]
#        cv2.imwrite("anchors1.jpg",cvimg)
        points,descript = self.Match.sift_feat(cvimg,[rect[0][0],rect[0][1],rect[2][0],rect[2][1]])
        self.points = points
        self.descripts = descript
        self.template_shape = rect[1][0] - rect[0][0]

    def match(self,img,template_dict):
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = template_dict['Rects']
        self.img_keypoints, self.img_descriptors = self.Match.sift_feat(self.gray_img)
        self.color_img = img
        maxsize  = max(img.shape[0],img.shape[1])
        img_aligned, alignMatrix = self.Match.post_match((100,100), self.points,
	                                                              self.descripts, self.img_keypoints,
	                                                              self.img_descriptors, self.color_img,
	                                                              input_roi=None,
	                                                              outputsize=(maxsize * 2,
	                                                                          maxsize * 2),
	                                                              offset=None)
#        cv2.imwrite("anchors.jpg", img_aligned)
        id_res = {}
        reslist = []
        for filed in template_dict['fields']:
            pts = filed['loc']
            for i in range(len(pts)):
                pts[i][0] = pts[i][0] - rect[0][0]
                pts[i][1] = pts[i][1] - rect[0][1]

            key = filed['name']
            type = filed['type']

            res = {}
            pattern = u"数字"
            if sys.version > '3':
                if type == pattern:
                    res[key] = self.recog.predict_digits(img_aligned, pts)
                if type == u"简体中文":
                    res[key] = self.recog.predict(img_aligned, pts)
            else:
                if unicode(type) == unicode(pattern):
                    res[key] = self.recog.predict_digits(img_aligned,pts)

                if unicode(type) == u"简体中文":
#                    print(pts)
                    res[key] = self.recog.predict(img_aligned, pts)
            reslist.append(res)

        id_res["status"] =0
        id_res['res'] = reslist


        return json.dumps(id_res,ensure_ascii=False)


__HOST = '0.0.0.0'
__PORT = 7000

class IDCardRecognizeHandler(object):
    def __init__(self):
        self.predictor = id_card()

    def recognize(self, img):
        return self.predictor.idcard_dect(img)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    handler = IDCardRecognizeHandler()
    print('Starting the rpc server at', __HOST,':', __PORT)

