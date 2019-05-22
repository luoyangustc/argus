#coding=utf-8
import cv2
from idcard_seg import IDCardSeg
from idcard_reco import idcard_reco
from idcard_post import idcard_post
import os
import json
import Levenshtein
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
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

    def idcard_dect_debug(self, img):
        img1, rect1, rect2 = self.id_seg.idcard_seg(img)
        preds = self.id_reco.predict(img1, rect1)
        json = self.id_post.postProcessing(preds)
        return img1,json

def parse_gt(gtfile):
    gt_result = {}
    #with open(gtfile, encoding='utf-8', errors='ignore') as f:
    with open(gtfile) as f:
        for line in f.readlines():
            gt = json.loads(line)
            imgfile = gt["img"]
            if gt["gt"]["status"] == -1:
                gt_result[imgfile] = {}
            else:
                gt_result[imgfile] = gt["gt"]["id_res"]

    return gt_result


if __name__ == '__main__':
    idcard = id_card()
    img_dir = "image_biaozhu"
    save_file = "result_baiozhu_4534_1227.txt"
    gt_file = "result_4534_gt.txt"
    gt_dict = parse_gt(gt_file)

    dis_sex = 0
    dis_number = 0
    dis_people = 0
    dis_name = 0
    dis_address = 0
    count_id=0
    count_all=0
    with open(save_file, 'w') as f:
         for imgfile in os.listdir(img_dir):
	     #print(imgfile)
             _imgfile = os.path.join(img_dir, imgfile)
             try:
                 result = idcard.idcard_dect(cv2.imread(_imgfile))
		 print(result)
             except:
                 continue
             result_dict = json.loads(result)['id_res']
             if imgfile in gt_dict.keys():
   		 print(imgfile)
                 gt_dict_this_img = gt_dict[imgfile.decode('utf-8')]
                 #print("{}:\n--result:{}\n--gt:{}\n".format(imgfile, result_dict, gt_dict_this_img))
                 for key in gt_dict_this_img:
                     if key == 'id_number':
                         dis_number += Levenshtein.distance(result_dict[key], gt_dict_this_img[key])
			 count_id = count_id + len(gt_dict_this_img[key])
                     if key == 'sex':
                         dis_sex += Levenshtein.distance(result_dict[key], gt_dict_this_img[key])
			 count_all = count_all + len(gt_dict_this_img[key])
                     if key == 'people':
                         dis_people += Levenshtein.distance(result_dict[key], gt_dict_this_img[key])
			 count_all = count_all + len(gt_dict_this_img[key])
                     if key == 'name':
                         dis_name += Levenshtein.distance(result_dict[key], gt_dict_this_img[key])
			 count_all = count_all + len(gt_dict_this_img[key])
                     if key == 'address':
                         dis_address += Levenshtein.distance(result_dict[key], gt_dict_this_img[key])
			 count_all = count_all + len(gt_dict_this_img[key])
             else:
                 print("{}\n--result:{}\n--gt: None\n".format(imgfile, result_dict))
         print("dis_number:" + str(float(dis_number) / count_id))
         print("dis_all:"+ str(float(dis_sex + dis_people + dis_name + dis_address)/(count_all)))




