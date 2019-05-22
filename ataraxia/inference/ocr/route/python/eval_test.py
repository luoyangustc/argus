from eval import *
#import eval
import unittest
deploy_path = "models/deploy.prototxt"
weight_caffemodel_path = "models/weight.caffemodel"
labels_csv = "models/labels.csv"
import json
import os


if __name__ == '__main__':
    configs = {
        "app": "cardapp",
        "use_device": "GPU",
        "batch_size":1,
        "custom_params":{
            "thresholds":  [0, 0.8, 0.8, 0.8, 0.8, 0.8,0.8]
        },
	    "model_files":{
            "deploy.prototxt":deploy_path,
            "weight.caffemodel":weight_caffemodel_path,
            "labels.csv":labels_csv,
        }
    }
    result_dict,_,_=create_net(configs)

    img_path = "examples/images_card/"
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