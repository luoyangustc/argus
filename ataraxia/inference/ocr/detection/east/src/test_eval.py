from eval import *
#import eval
import unittest
model_path = "/workspace/models/ocr-general-detection-201805111613/model.ckpt-58582.index"
import json
import os

if __name__ == '__main__':
    configs = {
        "app": "bkapp",
        "use_device": "GPU",
        "batch_size":256,
	"model_files":{
            "model.ckpt-58582.index":model_path,
        }
    }
    result_dict,_,_=create_net(configs)
    #net_preprocess(model, req)
    print(type(result_dict))
    img_path = "/workspace/EAST/20180513_new_test_pic/"
    img_list = os.listdir(img_path)
    reqs=[]
    temp_i = 0
    for img_name in img_list:
        reqs_temp = dict()
        reqs_temp["data"]=dict()
        reqs_temp["data"]["uri"]=img_path + img_name
        reqs_temp["data"]["body"]=None
        reqs.append(reqs_temp)
    ret = net_inference(result_dict, reqs)
    print(ret)

