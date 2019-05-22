# -*- coding: utf-8 -*-
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format
import time
from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


def change_deploy(deploy_file=None, input_data_batch_size=None):
    net = caffe_pb2.NetParameter()
    with open(deploy_file, 'r') as f:
        text_format.Merge(f.read(), net)
    data_layer_index = 0
    data_layer = net.layer[data_layer_index]
    data_layer.input_param.shape[0].dim[0] = input_data_batch_size
    with open(deploy_file, 'w') as f:
        f.write(str(net))


def parse_label_file(labelFile=None):
    label_dict = dict()
    with open(labelFile, 'r') as f:
        line_list = [i.strip() for i in f.readlines() if i]
        # Add by Riheng
        keyList = line_list[0].split(',')  # index,class,threshold
        for key in keyList[1:]:
            label_dict[key] = dict()
        for i_line in line_list[1:]:
            # Add by Riheng
            i_line_list = i_line.split(',')
            index_value = int(i_line_list[0])
            for colume_index, value in enumerate(i_line_list[1:], 1):
                label_dict[keyList[colume_index]][index_value] = value
    return label_dict


def decodeReqsImage(reqs):
    """
        process reqs image
        ----
        reqs :
        return :
            imageReadInfoDict,  # reqs image load info
            normalImageMapToOriginalImage, #  normal image id map to original image id
            images # load image result data
    """
    cur_batchsize = len(reqs)
    _t1 = time.time()
    imageReadInfoDict = dict()  # reqs image info
    normalImageMapToOriginalImage = dict()  # normal image id map to original image
    images = []
    normalImageIndex = 0
    for i in range(cur_batchsize):
        data = reqs[i]
        infoOfImage = dict()
        img = None
        try:
            # load image error
            img = load_image(data["data"]["uri"], body=data['data']['body'])
            if img is None:
                CTX.logger.info("input data is none : %s\n", data)
                infoOfImage['errorInfo'] = "image data is None"
                infoOfImage['errorCode'] = 400
                infoOfImage['flag'] = 1
            elif img.ndim != 3:
                CTX.logger.info("image ndim is " +
                                str(img.ndim) + ", should be 3\n")
                infoOfImage['errorInfo'] = "image ndim is " + \
                    str(img.ndim) + ", should be 3"
                infoOfImage['errorCode'] = 400
                infoOfImage['flag'] = 1
        except ErrorBase as e:
            CTX.logger.info(
                "image of index : %d,preProcess error: %s\n", i, str(e))
            infoOfImage['errorInfo'] = str(e)
            infoOfImage['errorCode'] = e.code
            infoOfImage['flag'] = 1  # 1 is  the image preprocess error
        if infoOfImage.get('flag', 0) == 1:  # the image preProcess error
            imageReadInfoDict[i] = infoOfImage
            continue
        # because , some images error, so need all images's map relation .
        infoOfImage['flag'] = 0  # normal image preProcess
        infoOfImage['normalImageIndex'] = normalImageIndex
        # new image id map to old image id
        normalImageMapToOriginalImage[normalImageIndex] = i
        imageReadInfoDict[i] = infoOfImage
        normalImageIndex += 1
        images.append(img)
    _t2 = time.time()
    CTX.logger.info("load images : %f\n", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)
    return imageReadInfoDict, normalImageMapToOriginalImage, images
