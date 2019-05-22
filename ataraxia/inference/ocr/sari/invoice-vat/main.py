#coding:UTF-8
import os
import cv2
import json
import base64
import numpy as np
import time
from PIL import Image
import traceback

from evals.src.invoice import ZenZhuiShui_reco
from evals.src.DaiKaiClassfier import clsPredictor_daikai
from evals.src.FiaoPiaoLianCiClassfier import clsPredictor_FiaoPiaoLianCi
from evals.src.postProcessDict import postProcess

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
        monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


@create_net_handler
def create_net(configs):
    daikai_model_path = os.path.join(os.path.dirname(__file__), 'src/models/DaiKai/cls.pkl')
    fapiaolian_model_path = os.path.join(os.path.dirname(__file__), 'src/models/LianCi/cls.pkl')
    # model = ZenZhuiShui_reco()
    daikai_model = clsPredictor_daikai(daikai_model_path)
    fapaiolian_model = clsPredictor_FiaoPiaoLianCi(fapiaolian_model_path)
    return {
        # "model": model, 
        "daikai_model": daikai_model, 
        "fapaiolian_model": fapaiolian_model, 
        "batch_size": configs['batch_size']
        }, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    # vat = model['model']
    daikai_model = model['daikai_model']
    fapaiolian_model = model['fapaiolian_model']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")

    ret = []
    cur_batchsize = len(reqs)
    if cur_batchsize > batch_size:
        raise ErrorOutOfBatchSize(batch_size)

    try:
        for req in reqs:
            if req["params"]["type"] == "detect":
                result = detect(req)
            elif req["params"]["type"] == "postrecog":
                result = postrecog(daikai_model, fapaiolian_model, req)
            else:
                return [], 400, 'bad request - with wrong params'
            
        ret.append({"code": 0, "message": "", "result": result})

    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)

    return ret, 0, ''


def detect(req):
    _t1 = time.time()
    # loading image
    img = load_image(req["data"]["uri"], body=req['data']['body'])
    _t2 = time.time()
    CTX.logger.info("inference :: cost for loading image: %f", _t2 - _t1)

    _t3 = time.time()
    vat = ZenZhuiShui_reco()    # create new instance
    rect_boxes,boxes_dict = vat.gen_img_dict(img)
    _t4 = time.time()

    CTX.logger.info("forward: %f", _t4 - _t3)
    monitor_rt_forward().observe(_t4 - _t1)
    return {
        "bboxes": rect_boxes,
        "dict": boxes_dict
    }


def postrecog(daikai_model, fapaiolian_model, req):
    img = load_image(req["data"]["uri"], body=req['data']['body'])
    # CTX.logger.info("load param - dict: %s", req["params"]["dict"])
    # CTX.logger.info("load param - texts: %s", req["params"]["texts"])
    # boxes_dict = json.loads(req["params"]["dict"])
    rec_result = req["params"]["texts"]
    CTX.logger.info("load param - texts: %s", rec_result)

    _t1 = time.time()
    vat = ZenZhuiShui_reco()    # 没法传递image_dict参数，只能新建实例然后重新生成参数
    _,boxes_dict = vat.gen_img_dict(img)
    # vat.gen_img_dict_base(img)
    vat.predict_oridinary(boxes_dict,rec_result)
    vat.predict_other(boxes_dict,rec_result)
    vat.predict_XiaoShouMingXi(boxes_dict,rec_result)
    vat.predict_svm(daikai_model)
    vat.predict_FaPiaoLianCi(fapaiolian_model)
    vat.predict_XiaoLeiMingCheng()
    res = postProcess(vat.out_dict)
    _t2 = time.time()

    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)

    return res


if __name__ == '__main__':
    # init model
    pass