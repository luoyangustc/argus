#-*- coding: utf-8 -*-
import cv2
from config import config
from crnnport import crnnSource, crnnRec_single
from digits_detect import digitDetect
from other import refine_boxes
class idcard_reco(object):
	def __init__(self):
		self.address_model, self.address_converter = crnnSource(config.RECOGNITION.ADDRESS_MODEL_PATH,
		                                                        config.RECOGNITION.ADDRESS_ALPHABET)
		self.name_model, self.name_converter = crnnSource(config.RECOGNITION.NAME_MODEL_PATH,
		                                                  config.RECOGNITION.NAME_ALPHABET)
		self.digits_model = digitDetect(config.RECOGNITION.DIGITS_MODEL_PATH,
										config.RECOGNITION.DIGITS_MIN_BIT,
										config.RECOGNITION.DIGITS_MAX_BIT)
		self.field_list = config.SEGMENT.TEMPLATE_FIELDS_LIST
		self.template_field_list = config.SEGMENT.TEMPLATE_FIELDS_LIST


	def predict(self, img, rects):
		predicts = []
		boxes=[]
		for idx, pt in enumerate(rects):
			left = pt[0][0]
			top = pt[0][1]
			right = pt[2][0]
			bottom = pt[2][1]
			boxes.append(left)
			boxes.append(top)
			boxes.append(right)
			boxes.append(bottom)
			field_img = img[top:bottom, left:right]
			if "address" in self.field_list[idx]:
				#print("address")
                		im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				cc,img1 = cv2.threshold(im_gray, 150,255, cv2.THRESH_BINARY_INV)
				x1_new,y1,x2_new,y2  = refine_boxes(img1,boxes)
				field_img1 = img[y1:y2,x1_new:x2_new]
				# rows,cols=field_img1.shape
				# for i in range(rows):
    			 	#for j in range(cols):
        		 	#    if (field_img1[i,j]==255):
           		 	#	    field_img1[i,j]=0
       			 	#	else:
            		 	#	    field_img1[i,j]=255
				f= 'image_address/'+str(idx) +'.jpg'
                		cv2.imwrite(f, img[y1:y2,x1_new:x2_new])
				field_predict = crnnRec_single(self.address_model, self.address_converter, field_img1, use_Threshold=True)
				#print("field_predict:"+str(field_predict))
				predicts.append(field_predict)
			elif "name" in self.field_list[idx]:
				#print("name")
				im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				cc,img1 = cv2.threshold(im_gray, 150,255, cv2.THRESH_BINARY_INV)
				x1_new,y1,x2_new,y2  = refine_boxes(img1,boxes)
				field_img1 = img[y1:y2,x1_new:x2_new]
				f= 'image_name/'+str(idx) +'.jpg'
                                cv2.imwrite(f, img[y1:y2,x1_new:x2_new])
				field_predict = crnnRec_single(self.name_model, self.name_converter, field_img1, use_Threshold=False)
				predicts.append(field_predict)
			elif "id" in self.field_list[idx]:
				#print("id")
				field_predict, probs = self.digits_model.digits_predict(field_img)
				#print(probs)
				predicts.append(field_predict)
				f= 'image_id/'+str(idx) +'.jpg'
                                cv2.imwrite(f, field_img)
				#field_predict = crnnRec_single(self.name_model, self.name_converter, field_img, use_Threshold=False)
                                #predicts.append(field_predict)
			else:
				#print("else situation")
				im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				cc,img1 = cv2.threshold(im_gray, 150,255, cv2.THRESH_BINARY_INV)
				x1_new,y1,x2_new,y2  = refine_boxes(img1,boxes)
				field_img1 = img[y1:y2,x1_new:x2_new]
				field_predict = crnnRec_single(self.name_model, self.name_converter, field_img1)
				predicts.append(field_predict)
			boxes=[]
		return predicts


if __name__ == '__main__':
	id_reg = idcard_reco()
