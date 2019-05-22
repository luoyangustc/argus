# -*- coding: utf-8 -*-
from crnnport import crnnSource, crnnRec_single

class TextRecognizer(object):
	def __init__(self, MODEL_FILE, MODEL_ALPHABET):
		self.model, self.converter = crnnSource(MODEL_FILE, MODEL_ALPHABET)

	def predict(self, img, rects):
		predicts = []
		for idx, rect in enumerate(rects):
			left = rect[0]
			top = rect[1]
			right = rect[2]
			bottom = rect[3]
			field_img = img[top:bottom, left:right]
			field_predict = crnnRec_single(self.model, self.converter, field_img, use_Threshold=True)
			predicts.append(field_predict)

		return predicts
