#coding=utf-8
import numpy as np
import cv2
import caffe
import argparse
import json
from cfg import Config

def init_models():
    if Config.PLATFORM == "GPU":
        caffe.set_mode_gpu()
        caffe.set_device(Config.TEST_GPU_ID)
    else:
        caffe.set_mode_cpu()
    # initialize the cls model
    cls_mod = caffe.Net(Config.CLS_NET_DEF_FILE,Config.CLS_MODEL_PATH,caffe.TEST)
    return cls_mod


def cls_process(net_cls, img):
    img = cv2.resize(img, (225, 225))
    img = img.astype(np.float32, copy=True)
    img -= np.array([[[103.94,116.78,123.68]]])
    img = img * 0.017
    img = img.transpose((2, 0, 1))
    net_cls.blobs['data'].data[...] = img
    out = net_cls.forward()
    score = out['prob'][0]
    sort_pre = sorted(enumerate(score) ,key=lambda z:z[1])
    label_cls = [sort_pre[-j][0] for j in range(1,2)]
    score_cls = [sort_pre[-j][1] for j in range(1,2)]
    return label_cls, score_cls



def save_json_in_text(json_dict=None, text_path=None):
    with open(text_path,'w') as f:
        json_result = json.dumps(json_dict, ensure_ascii=False)
        print json_result
        f.write(json_result)
        f.flush()
        f.close()
    pass


def process_image_fun(net_cls=None, image_path=None):
    cls_dict = {28: 'blog', 29: 'wechat', 30: 'other-text'}
    origimg = cv2.imread(image_path)
    if np.shape(origimg) != ():
        #starttime = time.time()
        label_cls, score_cls = cls_process(net_cls, origimg)
        #endtime = time.time()
        # print 'speed: {:.3f}s / iter'.format(endtime - starttime)
        json_dict={}
        cla_index = int(label_cls[0])
        if float(score_cls[0]) > Config.CLS_CONFIDENCE_THRESH and cla_index in cls_dict:
            json_dict['img']=image_path
            json_dict['img_type']=cls_dict.get(cla_index)
        else:
            json_dict['img']=image_path
            json_dict['img_type']="others"
        save_json_in_text(json_dict=json_dict,text_path=image_path+'_cls.json')


def parse_args():
    parser = argparse.ArgumentParser(description='AtLab Label Image!',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cls_mod = init_models()
    if args.image is not None:
        image_path = args.image
        process_image_fun(net_cls=cls_mod,image_path=image_path)
    else:
        with open(args.imagelist,'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            process_image_fun(net_cls=cls_mod,image_path=image_path)
