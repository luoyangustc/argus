# -*- coding: utf-8 -*-
import os
import os.path as osp

import json
# import base64
import time
import traceback
import hashlib
import struct

import urllib

# import cv2
import numpy as np

from evals.utils import CTX, create_net_handler
from evals.utils import net_inference_handler, net_preprocess_handler
from evals.utils import monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image

from face_aligner import FaceAligner
from mxnet_feature_extractor import MxnetFeatureExtractor


"""
请求：
{
    "data": {
        "uri": <uri:string>,   // 资源文件
        "attribute": {
            "pts":[[23,343],[23,434],[323,434],[323,343]] //每张图片每次只发一个脸框
        }
    },
    "params": {
        "roi_scale":<roi_scale:float>,  //可选，截图缩放比例，默认值：1.0
    }
}
返回：
{
    "code": 0,
    "message": "...",
    "body":  "feature binary stream"
}
"""

__LOCAL_PATH = osp.abspath(osp.dirname(__file__))
DEFAULT_APP_CONFIG_FNAME = osp.join(__LOCAL_PATH,
                                    'default_app_config.json')
DEFAULT_EXTRACTOR_CONFIG_FNAME = osp.join(__LOCAL_PATH,
                                          'default_extractor_config.json')


@create_net_handler
def create_net(configs):
    use_gpu = False
    roi_scale = 1.0

    CTX.logger.info("===> Input app configs: %s\n", str(configs))
    if not configs:
        configs = {}

    CTX.logger.info("===> Try to load default app configs from: %s and use them to update configs\n",
                    DEFAULT_APP_CONFIG_FNAME)
    try:
        fp = open(DEFAULT_APP_CONFIG_FNAME, 'r')
        _configs = json.load(fp)
        fp.close()
        CTX.logger.info("===> Loaded default app configs: %s\n", str(_configs))

        _configs.update(configs)
        configs = _configs

        CTX.logger.info("===> Updated app configs: %s\n", str(configs))

        mtcnn_model_path = ''
        feature_model_path = ''

        if("model_files" in configs):
            # print 'configs["model_files"]: ', configs["model_files"]
            for k, v in configs["model_files"].iteritems():
                if not mtcnn_model_path and k.startswith("mtcnn"):
                    if osp.isfile(v):
                        mtcnn_model_path = osp.dirname(v)
                    elif osp.isdir(v):
                        mtcnn_model_path = v
                if not feature_model_path and k.startswith("feature"):
                    if osp.isfile(v):
                        feature_model_path = osp.dirname(v)
                    elif osp.isdir(v):
                        feature_model_path = v

        if not mtcnn_model_path:
            raise Exception("Error: empty mtcnn_model_path\n")
        if not feature_model_path:
            raise Exception("Error: empty feature_model_path\n")

        configs["model_params"]["mtcnn_model_path"] = mtcnn_model_path
        configs["model_params"]["feature_model_path"] = feature_model_path
        configs["model_params"]["network_model"] = osp.join(
            feature_model_path, 'model,0')

        use_gpu = configs["use_device"].upper() == 'GPU'
        CTX.logger.info("===> use_gpu: %s", str(use_gpu))
        if 'gpu_id' not in configs["model_params"]:
            configs["model_params"]["gpu_id"] = 0

        if use_gpu:
            CTX.logger.info("===> gpu_id: %s", str(
                configs["model_params"]["gpu_id"]))

        if 'roi_scale' in configs["model_params"]:
            roi_scale = configs["model_params"]['roi_scale']

    except Exception as e:
        CTX.logger.error("Error when load and update app configs: %s\n",
                         traceback.format_exc())
        return {}, 521, str(e)

    CTX.logger.info("===> Updated app configs: %s\n", str(configs))

    CTX.logger.info("===> Try to load default extractor_config from: %s and update it by configs['model_params']\n",
                    DEFAULT_EXTRACTOR_CONFIG_FNAME)

    try:
        fp = open(DEFAULT_EXTRACTOR_CONFIG_FNAME, 'r')
        extractor_config = json.load(fp)
        fp.close()

        CTX.logger.info(
            "===> Loaded feature extractor configs: %s\n", str(extractor_config))

        if 'model_params' in configs:
            extractor_config.update(configs["model_params"])

        # if 'feature_model' in configs["model_params"]:
        #     extractor_config["network_model"] = configs["model_params"]["feature_model"]

        if 'batch_size' in configs:
            extractor_config["batch_size"] = configs["batch_size"]

        if use_gpu:
            extractor_config["cpu_only"] = False
        else:
            extractor_config["cpu_only"] = True
    except Exception as e:
        CTX.logger.error("Error when load and update extractor configs: %s\n",
                         traceback.format_exc())
        return {}, 522, str(e)

    CTX.logger.info("===> Updated feature extractor configs: %s",
                    str(extractor_config))

    try:
        feature_extractor = MxnetFeatureExtractor(extractor_config)
    except Exception as e:
        CTX.logger.error("Error when init face feature extractor: %s\n",
                         traceback.format_exc())

        return {}, 523, str(e)

    try:
        face_aligner = FaceAligner(str(configs["model_params"]["mtcnn_model_path"]),
                                   configs["model_params"]["gpu_id"] if use_gpu else -1)
    except Exception as e:
        CTX.logger.error("Error when init face feature extractor: %s\n",
                         traceback.format_exc())

        return {}, 524, str(e)

    model = {
        "feature_extractor": feature_extractor,
        "face_aligner": face_aligner,
        "batch_size": configs["batch_size"],
        "input_height": extractor_config["input_height"],
        "input_width": extractor_config["input_width"],
        "workspace": configs["workspace"],
        "roi_scale": roi_scale
    }

    return model, 0, 'Success'


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("===> PreProcess...\n")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    CTX.logger.info("===> Inference begin ...\n")

    try:
        face_chips = pre_eval(model, reqs)
        features = eval(model, face_chips)
        ret = post_eval(model, features, reqs)
    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("Inference error: %s\n", traceback.format_exc())
        return [], 599, str(e)

    return ret, 0, ''


def download_url(url, save=False):
    """Copy the contents of a file from a given URL
    to a local file.
    """
    web_file = urllib.urlopen(url)
    content = web_file.read()
    web_file.close()

    if save:
        local_file = open(url.split('/')[-1], 'w')
        local_file.write()
        local_file.close()

    return content


def check_req_data_body(req):
    body = req["data"].get('body', None)
    if not body:
        #-- if uri is a local file, 'body' can be omitted
        if osp.exists(req["data"]["uri"]):
            req["data"]["body"] = None
        else:  # -- if uri is a web url, try to download it
            CTX.logger.info("Download image: %s\n", req["data"]["uri"])
            try:
                img_data = download_url(req["data"]["uri"])
                req["data"]["body"] = img_data
            except Exception as e:
                CTX.logger.error("Error when download image: %s\n",
                                 traceback.format_exc())


def check_req_list(reqs):
    for req in reqs:
        check_req_data_body(req)


def check_landmarks(landmarks):
    t_array = np.array(landmarks)
    if t_array.size == 10:
        if t_array.shape == (5, 2):
            t_array = t_array.T
        t_array = t_array.reshape((10,))

        return t_array.tolist()
    else:
        CTX.logger.error(
            "Error: Wrong landmarks format, landmarks must be 5x2 or 10x1 list\n")
        raise ErrorInvalidPTS(landmarks)
        return None


def check_bbox(bbox, roi_scale=1.0):
    t_array = np.array(bbox)
    # print 'roi_scale:', roi_scale
    # print 'old bbox:', t_array

    if t_array.shape == (4, 2):
        t_array = t_array[(0,2), ...]
    elif t_array.size == 4:
        t_array = t_array.reshape((2, 2))

    # print 'old bbox:', t_array
    # print 't_array.shape:', t_array.shape

    if t_array.shape == (2, 2):
        if roi_scale != 1.0:
            ct_xy = (t_array[0] + t_array[1]) * 0.5
            old_wh = t_array[1] - t_array[0]

            new_wh_hf = old_wh * roi_scale * 0.5

            t_array = np.vstack((ct_xy - new_wh_hf, ct_xy + new_wh_hf))
            # print 'new bbox:', t_array
        return t_array.flatten().tolist()
    else:
        CTX.logger.error(
            "Error: Wrong bbox format, bbox must be 4x2 or 4x1 list\n")
        raise ErrorInvalidPTS(bbox)
        return None


def pre_eval(model, reqs):
    CTX.logger.info("---> Inference pre_eval() begin ...\n")
    # feature_extractor = model["feature_extractor"]
    face_aligner = model["face_aligner"]
    batch_size = model["batch_size"]

    output_square = model["input_height"] == model["input_width"]

    cur_batchsize = len(reqs)
    if cur_batchsize > batch_size:
        raise ErrorOutOfBatchSize(batch_size)

    face_chips = []
    _t1 = time.time()
    for i in range(cur_batchsize):
        # print reqs[i]
        CTX.logger.info('---> req[%d] image uri: %s',
                        i, reqs[i]["data"]["uri"])
        check_req_data_body(reqs[i])
        img = load_image(reqs[i]["data"]["uri"],
                         body=reqs[i]["data"]["body"])

        CTX.logger.info('---> req[%d] image shape: %s', i, str(img.shape))

        landmarks = None
        pts = None

        try:
            if "attribute" in reqs[i]["data"]:
                if "landmarks" in reqs[i]["data"]["attribute"]:
                    landmarks = reqs[i]["data"]["attribute"]["landmarks"]
                    CTX.logger.info(
                        '---> req[%d] face landmarks: %s', i, str(landmarks))
                    checked_landmarks = check_landmarks(landmarks)

                    _faces = face_aligner.get_face_chips(
                        img, [], [checked_landmarks], output_square=output_square)
                    face_chips.extend(_faces)

                elif "pts" in reqs[i]["data"]["attribute"]:
                    pts = reqs[i]["data"]["attribute"]["pts"]
                    roi_scale = model["roi_scale"]
                    if "params" in reqs[i] and "roi_scale" in reqs[i]["params"]:
                        roi_scale = reqs[i]["params"]["roi_scale"]
                        CTX.logger.info(
                            '---> req[%d] roi_scale: %s', i, str(roi_scale))

                    checked_pts = check_bbox(pts, roi_scale)

                    CTX.logger.info(
                        '---> req[%d] face bbox pts: %s', i, str(pts))
                    CTX.logger.info(
                        '---> req[%d] checked face bbox pts: %s', i, str(checked_pts))

                    _faces = face_aligner.get_face_chips(
                        img, [checked_pts], output_square=output_square)
                    face_chips.extend(_faces)

        except Exception as e:
            CTX.logger.error("Error when align and crop face: %s\n",
                             traceback.format_exc())
            raise Exception("Error align and crop face")

        if landmarks is None and pts is None:
            if (img.shape[0] == model["input_height"]
                    and img.shape[1] == model["input_width"]):
                face_chips.append(img)
            else:
                raise ErrorNoPTS(reqs[i]["data"]["uri"])

        # if pts is None:
        #     # if img.shape[0] != model["input_height"] or img.shape[1] != model["input_width"]:
        #     #     img = cv2.resize(
        #     #         img, (model["input_width"], model["input_height"]))
        #     #     face_chips.append(img)
        #         face_chips.append(img)
        # else:
        #         _faces = face_aligner.get_face_chips(
        #             img, [pts], output_square=output_square)
        #         face_chips.extend(_faces)

        # for i, chip in enumerate(face_chips):
        #     print "face chip #%d, shape: %s" % (i, str(chip.shape))
        #     win_name = "face_%d" % i
        #     cv2.imshow(win_name, chip)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    _t2 = time.time()
    CTX.logger.info(
        "===> Pre-eval Time (loading images and aligning faces): %f\n", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    return face_chips


def pack_feature_into_stream(feature):
    fmt = '>' + str(len(feature)) + 'f'
    stream = struct.pack(fmt, *feature)

    return stream


def unpack_feature_from_stream(stream):
    fmt = '>' + str(len(stream) / 4) + 'f'
    feature = struct.unpack(fmt, stream)

    return feature


def post_eval(model, features, reqs):
    '''
        parse net output, as numpy.mdarray, to EvalResponse
        Parameters
        ----------
        net: net created by net_init
        output: network numpy.mdarray
        reqs: parsed reqs from net_inference
        reqid: reqid from net_inference
        Return
        ----------
        resp: list of EvalResponse{
            "code": <code|int>,
            "message": <error message|str>,
            "body": <eval result|object>,
        }
    '''
    CTX.logger.info("---> Inference post_eval() begin ...\n")

    resp = []
    # reqid = None
    workspace = model["workspace"]

    _t1 = time.time()
    for i, feature in enumerate(features):
        #CTX.logger.debug("featuer len {} and feature {}".format(len(feature),feature),extra = {"reqid": reqid})
        # print '---> feature: ', feature
        np.save(osp.join(workspace, '%d.npy' % i), feature)
        # stream = struct.pack('>' + str(len(feature)) + 'f', *feature)
        stream = pack_feature_into_stream(feature)
        # print '---> packed stream: ', stream

        # feature_unpack = np.array(unpack_feature_from_stream(stream), np.float32)
        # print '---> unpacked feature from stream: ', feature_unpack
        # print '---> sum(feature-feature_unpack): ',
        # (feature-feature_unpack).sum()

        CTX.logger.info("struct.unpack info:" + ">" +
                        str(len(stream) / 4) + "f")

        #---- old response format (face-feature-v1, v2, v3)
        # hash_sha1 = hashlib.sha1()
        # hash_sha1.update(stream)
        # feature_file_name = os.path.join(workspace, hash_sha1.hexdigest())
        # file = open(feature_file_name, "wb")
        # file.write(stream)
        # file.close()
        # resp.append({"code": 0, "message": "",
        #               "result_file": str(feature_file_name)})

        #---- new response format (face-feature-v4)
        res = {
            "code": 0,
            "message": "",
            "result": {"uri": reqs[i]["data"]["uri"]},
            "body": stream
        }
        resp.append(res)

    _t2 = time.time()
    CTX.logger.info(
        "===> Post-eval Time (assembling responses): %f\n", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resp


def eval(model, face_chips):
    CTX.logger.info("---> Inference eval() begin ...\n")
    feature_extractor = model["feature_extractor"]
    # face_aligner = model["face_aligner"]
    batch_size = model["batch_size"]

    features = []
    _t1 = time.time()

    for i in range(0, len(face_chips), batch_size):
        _ftrs = feature_extractor.extract_features_batch(
            face_chips[i:i + batch_size])
        features.extend(_ftrs)

    _t2 = time.time()
    CTX.logger.info("===> Eval Time (Extracting features): %f\n", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)

    return features


if __name__ == '__main__':
    #    config_fn = 'demo_app_config_local.json'
    config_fn = 'demo_app_config.json'

    fp = open(config_fn, 'r')
    configs = json.load(fp)
    fp.close()

    save_dir = './rlt_features'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    print '\n===> create_net():'
    model, err_code, err_msg = create_net(configs)
    print "       model: ", model
    print "       err info: ", err_code, err_msg
    batch_size = model["batch_size"]
    print "       batch_size: ", batch_size

    reqs = [
        {
            "data": {
                "uri": r'C:\zyf\github\mxnet-feature-extractor-zyf\test_data\face_chips_112x112\1\1.jpg',
                "body": '',
                # "atrribute": <image_attribute>
            },
            # "params": <params>
        },
        {
            "data": {
                "uri": r'C:\zyf\github\mxnet-feature-extractor-zyf\test_data\face_chips_112x112\1\2.jpg',
                "body": '',
                # "atrribute": <image_attribute>
            },
            # "params": <params>
        }
    ]

    print '\n===> net_inference():'
    batch_cnt = 0
    req_cnt = 0
    for i in range(0, len(reqs), batch_size):
        batch_cnt += 1
        print '\n---> process batch #%d' % batch_cnt
        resp = net_inference(model, reqs[i:i + batch_size])
        print '\n---> inference result: ', resp

        for res in resp[0]:
            req_cnt += 1
            print '\n---> feature #%d' % req_cnt
            # print '\n---> inference result: ', resp

            #---- old response format (face-feature-v1, v2, v3)
            # with open(res["result_file"], 'rb') as fp:
            #     stream = fp.read()
            #     feature_unpack = np.array(unpack_feature_from_stream(stream), np.float32)
            #     print '---> unpacked feature from stream: ', feature_unpack
            #     save_fn = osp.join(save_dir, 'feat_req_%d.npy' % req_cnt)
            #     np.save(save_fn, feature_unpack)

            #---- new response format (face-feature-v4)
            stream = res["body"]
            feature_unpack = np.array(
                unpack_feature_from_stream(stream), np.float32)
            print '---> unpacked feature from stream: ', feature_unpack
            save_fn = osp.join(save_dir, 'feat_req_%d.npy' % req_cnt)
            np.save(save_fn, feature_unpack)
