import sys
import os.path as osp
import json

import numpy as np

from binary_stream_ops import pack_feature_into_stream
from eval_face_cluster import create_net, net_inference


def assemble_request_from_feat_list(feat_list_fn, root_dir=None, endian='big'):
    req_datas = []
    big_endian = endian.lower() == 'big'

    if not root_dir:
        root_dir = "./"

    print "feat_list_fn: ", feat_list_fn
    print "big_endian: ", big_endian
    print "root_dir: ", root_dir

    with open(feat_list_fn, 'r') as fp:
        for line in fp:
            feat_fn = osp.join(root_dir, line.strip())
            if feat_fn.endswith('.npy'):
                feat = np.load(feat_fn)
                # print "feat.dtype", feat.dtype
                feat_stream = pack_feature_into_stream(feat, big_endian)
            else:
                with open(feat_fn, 'rb') as fp_ft:
                    feat_stream = fp_ft.read()

            req_data_i = {
                "uri": line,
                "body": feat_stream,
                "attribute": {
                    "face_id": "null",
                    "group_id": -2
                }
            }

            req_datas.append(req_data_i)

    request_dict = {"datas": req_datas}
    # print "request_dict: ", request_dict

    return request_dict


if __name__ == '__main__':
    config_fn = 'create_params.json'
    feat_list_fn = './test_feature_bin_list.txt'

    if len(sys.argv) > 1:
        feat_list_fn = sys.argv[1]
    if len(sys.argv) > 2:
        config_fn = sys.argv[2]

    fp = open(config_fn, 'r')
    configs = json.load(fp)
    fp.close()
    print '\n===> configs:', configs

    print '\n===> create_net():'
    model, err_code, err_msg = create_net(configs)
    print "       model: ", model
    print "       err info: ", err_code, err_msg

    print '\n===> assemble request:'

    req_1 = assemble_request_from_feat_list(feat_list_fn)

    reqs = [req_1]

    print '\n===> net_inference():'
    resp = net_inference(model, reqs, "face_cluster:infer")
    print '\n---> inference result: ', resp
