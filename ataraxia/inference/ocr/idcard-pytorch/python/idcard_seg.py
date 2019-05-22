# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from config import config

DEBUG = False

class IDCardMatcher:
	def __init__(self):
		self.sift = cv2.xfeatures2d.SIFT_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
		search_params = dict(checks=10)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def sift_feat(self, img, input_roi=None):
		#print(img)
		if input_roi != None:
			_img = img[input_roi[1]:input_roi[3], input_roi[0]:input_roi[2]]
		else:
			_img = img
		return self.sift.detectAndCompute(_img, None)

	def post_match(self, template_shape, kp1, des1, kp2, des2, color_img, input_roi=None,
				   outputsize=None, offset=None):
		matches = self.flann.knnMatch(des1, des2, k=config.SEGMENT.POST_MATCH.knn_match_k)
		threshold = config.SEGMENT.POST_MATCH.threshold
		good_match_num = config.SEGMENT.POST_MATCH.good_match_num
		good_matches = []
		for m, n in matches:
			if m.distance < threshold * n.distance:
				good_matches.append(m)
		#print(good_matches)
		if len(good_matches) > good_match_num:
			src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
			dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
			if input_roi != None:
				dst_pts = np.float32(dst_pts + [input_roi[0], input_roi[1]])
			dst_pts = dst_pts.reshape(-1, 1, 2)
			M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
			if offset != None:
				M = np.dot(np.float32([[1, 0, -offset[0]], [0, 1, -offset[1]], [0, 0, 1]]), M)
			matchesMask = mask.ravel().tolist()
			h, w = template_shape
			pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

			if outputsize == None:
				outputsize = (color_img.shape[0],color_img.shape[1])

			outputsize = (int(outputsize[0]),int(outputsize[1]))
			return cv2.warpPerspective(color_img, M, outputsize), np.linalg.inv(M)  # inv(M) M-1 warPerspective
		else:
			return None, None  # M


class IDCardSeg:
	def __init__(self):
		self.idcard_matcher = IDCardMatcher()
		self.index= 0
		segment_template = cv2.imread(config.SEGMENT.TEMPLATE_IMG, cv2.IMREAD_GRAYSCALE)
		self.template_keypoints, self.template_descriptors = self.idcard_matcher.sift_feat(segment_template)
		self.template_shape = segment_template.shape
		self.template_field = config.SEGMENT.TEMPLATE_FIELDS
		self.template_field_list = config.SEGMENT.TEMPLATE_FIELDS_LIST

	def _seg(self, img):
		self.gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		self.img_keypoints,self.img_descriptors = self.idcard_matcher.sift_feat(self.gray_img)
		if len(self.img_keypoints) == 0:
			return None, None, None, None
		self.color_img = img
		img_aligned, alignMatrix = self.idcard_matcher.post_match(self.template_shape, self.template_keypoints,
																  self.template_descriptors,self.img_keypoints,
																  self.img_descriptors, self.color_img,
																  input_roi = None,
																  outputsize=(self.template_shape[1] * 2,
																              self.template_shape[1] * 2),
																  offset=None)
		if img_aligned is None and alignMatrix is None:
			return None, None, None, None

		if img_aligned.shape[0] < self.template_shape[0] or img_aligned.shape[1] < self.template_shape[1]:
			return None,None,None,None


		coords_aligned = []
		coords_list =[]
		field_imgs = []
		for name in self.template_field_list:
			coords_list.append(self.template_field[name])
			coords_aligned.extend(self.template_field[name])
			if DEBUG:
				left = self.template_field[name][0][0]
				top = self.template_field[name][0][1]
				right = self.template_field[name][3][0]
				bottom = self.template_field[name][3][1]
				field_imgs.append(img_aligned[top:bottom,left:right])

		coords_original = np.vstack((np.array(coords_aligned).T,np.ones((1,len(coords_aligned)))))
		coords_original = np.dot(alignMatrix,coords_original)
		coords_original[0,:] /= coords_original[2,:]
		coords_original[1,:] /= coords_original[2,:]
		return img_aligned, coords_list, field_imgs, coords_original[0:2,:]

	def idcard_seg(self, img):

		img_aligned, coords_aligned, field_imgs, coords_original = self._seg(img)

		if DEBUG:
			for idx, name in enumerate(self.template_field_list):
				if not os.path.exists('./IDImage'+"_"+name):
					os.mkdir('./IDImage'+"_"+name)
				f= './IDImage'+"_"+name+'/IDCARD_'+"2"+'_'+ str(idx) +'.jpg'
				cv2.imwrite(f, field_imgs[idx])

		return img_aligned, coords_aligned, coords_original


if __name__ == "__main__":
	idcard_seg = IDCardSeg()
	for f in os.listdir('./IDCARD/'):
		idcard_seg.idcard_seg('./IDCARD/'+f)
