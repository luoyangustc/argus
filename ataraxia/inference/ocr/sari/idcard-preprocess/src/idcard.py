
import json
import requests
import config
import time
import base64
import logging
from src.IDCard.idcard_class import IDCARDCLASS
import config
import cv2
import os
import numpy as np
from src.IDCard.IDCardfront.Aligner import AlignerIDCard
from src.IDCard.IDCardback.Aligner import AlignerIDCardBack

class IDCard(object):
    def __init__(self):
        self.idcardcls = IDCARDCLASS(config.ALIGNER_TEMPLATE_IDCARDCLASS_IMG_PATH,config.ALIGNER_TEMPLATE_IDCARDCLASSBACK_IMG_PATH)
        
        self.idcardfront = AlignerIDCard(config.ALIGNER_TEMPLATE_IDCARD_IMG_PATH,config.ALIGNER_TEMPLATE_IDCARD_LABEL_PATH)
        self.idcardback = AlignerIDCardBack(config.ALIGNER_TEMPLATE_IDCARDBACK_IMG_PATH,config.ALIGNER_TEMPLATE_IDCARDBACK_LABEL_PATH)

    def idcard_recog(self,img):
        image = cv2.imdecode(np.fromstring(base64.b64decode(img), dtype=np.uint8), 1)
        cls = self.idcardcls.run(image)
       
        if(cls == 0):
            proc_handler = self.idcardfront
            res_default = json.dumps({'公民身份号码':'', '性别':'', '民族':'','出生':'','住址':'', '姓名':''})
        else:
            proc_handler = self.idcardback
            res_default = json.dumps({'有效期限':'', '签发机关':''})
    
        try:
            # res = self.idcardfront.run(img,num)
            # pre-process
            alignedImg,names,regions,boxes = proc_handler.predet(img)
            # detection
            detectedBoxes = proc_handler.det(alignedImg)
            # pre-recognition
            boxes = proc_handler.prerecog(detectedBoxes,alignedImg,names,regions,boxes)
            # recognition
            texts = proc_handler.rec(alignedImg,boxes)
            # post-process
            res = proc_handler.postprocess(boxes,texts,regions,names)

            return json.dumps(res)
        except:
            return res_default

if __name__ == '__main__':
    # 测试使用entry.py
    pass