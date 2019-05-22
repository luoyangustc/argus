# -*- coding: utf-8 -*-
import os
import os.path as osp

import json
# import base64
import time
import traceback

from evals.utils import CTX, create_net_handler
from evals.utils import net_inference_handler, net_preprocess_handler
from evals.utils import monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *

from url_ops import download_url
from binary_stream_ops import unpack_feature_from_stream

from facex_cluster import incrementally_cluster, cluster_all_features


"""
请求：
{
    "datas": [
        {   // 第一个人脸特征信息
            "uri": <string>,            // 与body字段可二选一，face feature二进制文件的uri，
                                        // 如body字段有效，则忽略改uri
            "attribute": {
                "face_id": <string>     // 人脸id
                "group_id": <int>       // 前一次的聚类标签, 该字段为空或者-2，表示未参与过前一次聚类；
                                        // =-1，表示前一次未聚成功；大于等于0，表示前一次的聚类编号
            }
            "body": <binary stream>     // 与uri字段可二选一，人脸特征，二进制流
        },
        {   // 第二个人脸特征信息
            "uri": <string>,            // 与body字段可二选一，face feature二进制文件的uri，
                                        // 如body字段有效，则忽略改uri
            "attribute": {
                "face_id": <string>     // 人脸id
                "group_id": <int>       // 前一次的聚类标签, 该字段为空或者-2，表示未参与过前一次聚类；
                                        // =-1，表示前一次未聚成功；大于等于0，表示前一次的聚类编号
            }
            "body": <binary stream>     // 与uri字段可二选一，人脸特征，二进制流
        },
        ...
    ],
    "params": {
        "sim_thresh": <float>,   //可选，聚类阈值(相似度，非阈值)参数，可在初始化config中设置，默认值：0.45
        "min_samples_per_cluster": <int>,  //可选，每个聚类的最少格式，可在初始化config中设置，默认值：2
        "endian": <string>,     //可选，特征二进制流的endian，取值只能为"big"或者"little",
                                //可在初始化config中设置，默认值："big"
        "incrementally": <int>    //可选，是否增量聚类，取值0或1，可在初始化config中设置，默认为1
    }
}

返回：
{
    "code": 0,
    "message": "...",
    "result": <cluster result： json string>
}

<cluster result: json string>:
    [
        {
            "face_id": <string>,
            "group_id": <int>,
            "distance_to_center": <float>
        },
        {
            "face_id": <string>,
            "group_id": <int>
            "distance_to_center": <float>
        },
        ...
    ]
"""


@create_net_handler
def create_net(configs):
    model = {
        "sim_thresh": 0.55,
        "endian": "big",
        "incrementally": True,
        "min_samples_per_cluster": 2
    }

    CTX.logger.info("===> Default cluster params: %s\n",
                    str(model))

    if not isinstance(configs, dict):
        msg = 'init "configs" must be a dict type'
        CTX.logger.error(msg)
        return {}, 521, str(msg)

    if 'custom_params' in configs:
        try:
            if 'sim_thresh' in configs["custom_params"]:
                thresh = float(configs["custom_params"]["sim_thresh"])
                if thresh < 1.0 and thresh > 0.0:
                    model["sim_thresh"] = thresh
                else:
                    msg = 'configs["custom_params"]["sim_thresh"] must be in (0.0, 1.0)'
                    CTX.logger.error(msg)
                    return {}, 521, str(msg)

            if 'min_samples_per_cluster' in configs["custom_params"]:
                min_samples = int(configs["custom_params"]
                                  ["min_samples_per_cluster"])
                if min_samples > 1:
                    model["min_samples_per_cluster"] = min_samples
                else:
                    msg = 'configs["custom_params"]["min_samples_per_cluster"] must be > 1'
                    CTX.logger.error(msg)
                    return {}, 521, str(msg)

            if 'endian' in configs["custom_params"]:
                endian = configs["custom_params"]["endian"].lower()
                if endian not in ['big', 'little']:
                    msg = 'configs["custom_params"]["endian"] must be "big" or "little"'
                    CTX.logger.error(msg)
                    return {}, 521, str(msg)
                else:
                    model["endian"] = configs["custom_params"]["endian"]

            if 'incrementally' in configs["custom_params"]:
                model["incrementally"] = not(
                    not(configs["custom_params"]["incrementally"]))
        except Exception as e:
            CTX.logger.error("Error when load and update init configs: %s\n",
                             traceback.format_exc())
            return {}, 522, str(e)

    CTX.logger.info("===> Updated init configs: %s",
                    str(model))

    return model, 0, 'Success'


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("===> PreProcess not implemented...\n")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    CTX.logger.info("===> Inference begin ...\n")

    results = []
    for i, req in enumerate(reqs):
        CTX.logger.info("---> Process request #{} ...\n".format(i))
        # CTX.logger.info("---> Request content: {}".format(req))

        suc, msg, features = pre_eval(model, req)
        CTX.logger.info("pre_eval() return msg: " + msg)
        # CTX.logger.info("pre_eval() return features: " + str(features))

        if suc:
            eval_rlt = eval(model, features)
        else:
            eval_rlt = {
                "code": 522,
                "message": msg,
                "result": ""
            }

        results.append(eval_rlt)

    return results, 0, ''


def pre_eval(model, req):
    CTX.logger.info("---> Inference pre_eval() begin ...\n")
    # feature_extractor = model["feature_extractor"]
    msg = 'success'

    big_endian = True if model.get('endian', 'big') == 'big' else False

    if 'datas' not in req:
        msg = "No 'datas' field found in request"
        CTX.logger.error(msg)

        return False, msg, []

    if len(req["datas"]) < 2:
        msg = "Must have len(req['datas']) > 1"
        CTX.logger.error(msg)

        return False, msg, []

    _t1 = time.time()

    features = []

    last_feat_len = 0
    for req_data in req['datas']:
        body = req_data.get('body', None)

        if not body:
            # -- if uri is a local file, 'body' can be omitted
            if osp.exists(req_data["uri"]):
                CTX.logger.info(
                    "Try to load feature from local file: %s\n" % req_data["uri"])
                try:
                    # if req_data["uri"].endswith('.npy'):
                    #     feat_npy=np.load(req_data["uri"]).flatten()
                    #     feat_len=len(feat_npy)
                    #     req_data["feature"]=feat_npy
                    # else:
                    with open(req_data["uri"], 'rb') as fp:
                        req_data["body"] = fp.read()
                except Exception as e:
                    msg = 'Error when loading feature from local file: ' + traceback.format_exc()
                    CTX.logger.error(msg)
                    return False, msg, []
            else:  # -- if uri is a web url, try to download it
                CTX.logger.info("Download feature file: %s\n", req_data["uri"])
                try:
                    feat_data = download_url(req_data["uri"])
                    req_data["body"] = feat_data
                except Exception as e:
                    msg = 'Error when downloading feature from URI: ' + traceback.format_exc()
                    CTX.logger.error(msg)
                    return False, msg, []

        body = req_data.get('body', None)  # update body
        if not body:
            msg = 'all elements in req["datas"] must have valid "body" field or valid "uri" field'

            return False, msg, []
        else:
            try:
                feat_len, feat_npy = unpack_feature_from_stream(
                    body, big_endian)
            except Exception as e:
                msg = 'Error when unpacking one of input features: ' + traceback.format_exc()
                CTX.logger.error(msg)
                return False, msg, []

            if last_feat_len <= 0:
                last_feat_len = feat_len

            if feat_len < 1 or feat_len != last_feat_len:
                msg = 'invalid unpacked feature length. (feat_len=%d vs. last_feat_len=%d)' % (
                    feat_len, last_feat_len)
                CTX.logger.error(msg)
                return False, msg, []
            else:
                attr = req_data.get('attribute', {})
                face_id = attr.get('face_id', 'null')
                group_id = attr.get('group_id', -2)

                gt_id = attr.get('gt_id', -1)

                tmp = {
                    "face_id": face_id,
                    "feature": feat_npy,
                    "group_id": group_id,
                    "gt_id": gt_id
                }

                features.append(tmp)

    _t2 = time.time()
    CTX.logger.info(
        "===> Pre-eval Time (loading images and aligning faces): %f\n", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    return True, msg, features


def eval(model, features):
    CTX.logger.info("---> Inference eval() begin ...\n")
    _t1 = time.time()

    inter_index = "_inter_index"

    min_samples = model['min_samples_per_cluster']
    dist_thresh = 1.0 - model['sim_thresh']

    clustered_data_list = []
    single_data_list = []
    new_data_list = []
    max_gid = -1

    all_ft_list = features

    # print "features: ", features

    for i in range(len(all_ft_list)):
        all_ft_list[i][inter_index] = i

    # calc_feature_list_norm_inv(all_ft_list)
    # print "all_ft_list: ", all_ft_list

    for ft in all_ft_list:
        if 'feature' not in ft or len('feature') < 1:
            continue

        if model['incrementally']:
            if ft['group_id'] >= 0:
                clustered_data_list.append(ft)
            elif ft['group_id'] == -1:
                single_data_list.append(ft)
            else:
                new_data_list.append(ft)

            if ft['group_id'] > max_gid:
                max_gid = ft['group_id']
        else:
            new_data_list.append(ft)

    rlt_list = all_ft_list

    if len(new_data_list) > 0:
        if len(clustered_data_list) < 1:
            (rlt_clustered_list, rlt_single_list) = cluster_all_features(
                all_ft_list, dist_thresh, min_samples)

        else:
            (rlt_clustered_list, rlt_single_list) = incrementally_cluster(clustered_data_list,
                                                                          single_data_list, new_data_list,
                                                                          dist_thresh, min_samples, max_gid)
        rlt_list = rlt_clustered_list + rlt_single_list
    else:
        for item in single_data_list:
            item['group_id'] = -1
            item['distance_to_center'] = 0.0
        for item in clustered_data_list:
            item['distance_to_center'] = 0.0
        rlt_list = clustered_data_list + single_data_list

    # indices = [ft[inter_index] for ft in rlt_list]
    # new_rlt_list = [rlt_list[i] for i in indices]

    cluster_rlt = []
    for ft in rlt_list:
        tmp = {
            'face_id': ft['face_id'],
            'group_id': ft['group_id'],
            'distance_to_center':  round(ft['distance_to_center'], 6)
        }

        cluster_rlt.append(tmp)

    _t2 = time.time()

    CTX.logger.info(
        "===>clustering {} features costs {} seconds\n".format(
            len(features), _t2 - _t1))
    monitor_rt_post().observe(_t2 - _t1)

    dict_info = {
        "code": 0,
        "message": "",
        "result": json.dumps(cluster_rlt)
    }

    return dict_info


if __name__ == '__main__':
    config_fn = 'create_params.json'

    fp = open(config_fn, 'r')
    configs = json.load(fp)
    fp.close()
    print '\n===> configs:', configs

    print '\n===> create_net():'
    model, err_code, err_msg = create_net(configs)
    print "       model: ", model
    print "       err info: ", err_code, err_msg

    req_1 = {
        "datas": [
            {
                "uri": "",
                "body": "",
                "attribute": {
                    "face_id": "person_1",
                    "group_id": -2
                }
            },
            {
                "uri": "",
                "body": "",
                "attribute": {
                    "face_id": "person_2",
                    "group_id": -2
                }
            }
        ],
        "params": {}
    }

    reqs = [req_1]

    print '\n===> net_inference():'
    resp = net_inference(model, reqs, "face_cluster:infer")
    print '\n---> inference result: ', resp
