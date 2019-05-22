#coding:UTF-8
import os
import cv2
import json
import base64
import numpy as np
import requests
from PIL import Image
from Coarse2Finecut import Coarse2Fine_cut
import KeyList


class ZenZhuiShui_reco():
    def __init__(self):
        template_path = os.path.join(os.path.dirname(__file__), 'data/template')
        # daikai_model_path = os.path.join('src','zengzhishui','models', 'DaiKai', 'cls.pkl')
        # fapiaolian_model_path = os.path.join('src','zengzhishui','models', 'LianCi', 'cls.pkl')
        self.coarse2fine_cut = Coarse2Fine_cut(template_path)
        # self.daikai_model = clsPredictor_daikai(daikai_model_path)
        # self.fapaiolian_model = clsPredictor_FiaoPiaoLianCi(fapiaolian_model_path)
        self.initOutDict()

    # def deploy(self,img,rect_boxes):
    #     img_encoded = base64.b64encode(cv2.imencode('.png',img)[1].tostring())
    #     r = requests.post(self.address, data=json.dumps({'img':img_encoded.decode('utf-8'),'bboxes':rect_boxes}))
    #     result = json.loads(r.text)
    #     print(result)
    #     return result

    def initOutDict(self):
        self.out_dict = {}
        for key in KeyList.key_list:
            self.out_dict[key] = ''

    def predict_oridinary(self,rect_boxes,rec_result):
        for key in KeyList.oridinary_key_list:
            if self.imgs_dict[key] != []:
                predict = ''
                for i in range(len(self.imgs_dict[key])):
                    input = self.imgs_dict[key][i]
                    # print(rect_boxes[key][i][0],rect_boxes[key][i][1])
                    predict += ' '.join(rec_result[rect_boxes[key][i][0]:rect_boxes[key][i][1]])
                self.out_dict[key] = predict

    def predict_other(self,rect_boxes,rec_result):
        for key in KeyList.other_key_list:
            if self.imgs_dict[key] != []:
                predict = ''
                for i in range(len(self.imgs_dict[key])):
                    input = self.imgs_dict[key][i]
                    predict += ' '.join(rec_result[rect_boxes[key][i][0]:rect_boxes[key][i][1]])
                self.out_dict[key] = predict

    def predict_XiaoShouMingXi(self,rect_boxes,rec_result):
        '''
        这里比较特殊，返回一个二维数组
        :return:
        '''
        out_key = '_XiaoShouMingXi'
        ret = [[[] for j in range(8)] for i in range(8)]
        result = [[[] for j in range(8)] for i in range(8)]
        for key in self.imgs_dict.keys():
            if 'Table' in key:
                col = int(key[-1]) - 1
                for row, line in enumerate(self.imgs_dict[key]):
                    ret[row][col] = line  # row是一个1维list
                    
                    result[row][col] = rec_result[rect_boxes[key][row][0]:rect_boxes[key][row][1]]

        rows = 0
        flag = 1
        for i in range(len(ret)):
            for j in [0, 5, 6, 7]:
                if ret[i][j] == []:
                    rows = i
                    flag = 0
                    break
            if flag == 0:
                break

        ret1 = [['' for j in range(8)] for i in range(rows)]

        for i in range(rows):
            for j in range(8):
                if ret[i][j] != []:
                    ret1[i][j] = ' '.join(result[i][j])

        self.out_dict[out_key] = ret1

    def predict_svm(self, daikai_model):
        for key in KeyList.to_cls_key_list:
            if key == '_DaiKaiBiaoShi':
                if self.imgs_dict[key] != []:
                    predict = daikai_model.predict(
                        self.imgs_dict[key][0][0])[0]

                    if predict.decode('utf-8') == 'daikai' or u'代开' in self.out_dict['_BeiZhu']\
                        or u'代开' in self.out_dict['_XiaoShouFangMingCheng']\
                        or u'代开' in self.out_dict['_XiaoShouFangNaShuiRenShiBieHao']\
                        or u'代开' in self.out_dict['_XiaoShouFangDiZhiJiDianHua']\
                        or u'代开' in self.out_dict['_XiaoShouFangKaiHuHangJiZhangHao']:
                        self.out_dict[key] = u'代开'

    def predict_FaPiaoLianCi(self, fapaiolian_model):
        key = '_FaPiaoLianCi'
        im = self.imgs_dict[key][0][0].copy()
        predict = fapaiolian_model.predict(im)[0]
        if predict == 'FaPiaoLian':
            self.out_dict[key] = u'发票联'
        elif predict == 'DiKouLian':
            self.out_dict[key] = u'抵扣联'
        else:
            self.out_dict[key] = u'记账联'

    # 必须在最后预测
    def predict_XiaoLeiMingCheng(self):
        key = '_XiaoLeiMingCheng'
        if self.out_dict['_DaiKaiBiaoShi'] == u'代开':
            self.out_dict[key] = u'增值税代开发票'
        elif u'专用' in self.out_dict['_DanZhengMingCheng']:
            self.out_dict[key] = u'一般增值税发票'
        else:  # u'普通' in self.out_dict['_DanZhengMingCheng']:
            self.out_dict[key] = u'增值税普通发票'

    # def predictAndGenXML(self, img):
    #     self.initOutDict()
    #     # root = ET.Element("root")     
    #     rect_boxes,boxes_dict = self.gen_img_dict(img)
    #     # print(rect_boxes)
    #     rec_result = self.deploy(img,rect_boxes)
        
    #     self.predict_oridinary(boxes_dict,rec_result)
    #     self.predict_other(boxes_dict,rec_result)
    #     self.predict_XiaoShouMingXi(boxes_dict,rec_result)
    #     # cv2.imshow('image',img)
    #     # cv2.waitKey(0)
    #     self.predict_svm(daikai_model)
    #     self.predict_FaPiaoLianCi(fapaiolian_model)
    #     self.predict_XiaoLeiMingCheng()
    #     self.out_dict = postProcess(self.out_dict)
    #     json_content = json.dumps(self.out_dict)
    #     return json_content # xml

    def gen_img_dict(self, img):
        self.gen_img_dict_base(img)
        rect_boxes = []
        boxes_dict = {}
        for key in self.coords_dict:
            _2dboxes = self.coords_dict[key]
            boxes_dict[key] = {}
            for idx_2d,_1dboxes in enumerate(_2dboxes):
                boxes_dict[key][idx_2d] = [len(rect_boxes),len(rect_boxes)+len(_1dboxes)]
                for box in _1dboxes:
                    rect_boxes.append(box.tolist())
        return rect_boxes,boxes_dict

    def gen_img_dict_base(self, img):
        self.imgs_dict, self.coords_dict = self.coarse2fine_cut.cut(img)
