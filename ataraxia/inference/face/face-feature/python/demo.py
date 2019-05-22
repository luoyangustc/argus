import os
import os.path as osp
import numpy as np

import json

from eval_face_features import create_net, net_inference, unpack_feature_from_stream
from compare_feats import compare_feats


if __name__ == '__main__':
    # configs = {
    #     'app': 'face-feature'
    # }
    #    config_fn = 'demo_app_config_local_r100.json'
    #    config_fn = 'demo_app_config_local.json'
    config_fn = 'demo_app_config.json'
    # req_url_fn = 'test_json_list/test_urls_2pts_float.json'
    # req_url_fn = 'test_json_list/test_urls_2pts_int.json'
    req_url_fn = 'test_json_list/test_urls_4pts_int.json'
    # req_url_fn = 'test_json_list/test_urls_4pts_int_with_param.json'
    # req_url_fn = 'test_json_list/test_urls_4pts_int_with_landmarks.json'

    fp = open(config_fn, 'r')
    configs = json.load(fp)
    fp.close()
    print '\n===> Input app configs: ', configs

    fp = open(req_url_fn, 'r')
    reqs = json.load(fp)
    fp.close()

    save_dir = './rlt_features_demo'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fn_resp = osp.join(save_dir, 'rlt_response.txt')
    fp_resp = open(fn_resp, 'w')

    fn_feat = osp.join(save_dir, 'rlt_feat_stream.txt')
    fp_feat = open(fn_feat, 'w')

    fn_feat2 = osp.join(save_dir, 'face_feature_regtest.tsv')
    fp_feat2 = open(fn_feat2, 'w')

    print '\n===> create_net():'
    model, err_code, err_msg = create_net(configs)
    print "       model: ", model
    print "       err info: ", err_code, err_msg
    batch_size = model['batch_size']
    print "       batch_size: ", batch_size

    print reqs

    batch_cnt = 0
    req_cnt = 0

    for i in range(0, len(reqs), batch_size):
        batch_cnt += 1
        print '\n===> process batch #%d' % batch_cnt
        print '\n===> net_inference():'
        resp = net_inference(model, reqs[i:i + batch_size])
        print '\n===> inference result: ', resp

        if len(resp[0]) > 0:
            for res in resp[0]:
                req_cnt += 1
                print '\n---> feature cnt: %d' % req_cnt

                #---- old response format (face-feature-v1, v2, v3)
                # with open(res['result_file'], 'rb') as fp:
                #     stream = fp.read()
                #     feature_unpack = np.array(unpack_feature_from_stream(stream), np.float32)
                #     print '---> unpacked feature from stream: ', feature_unpack
                #     save_fn = osp.join(save_dir, 'feat_req_%d.npy' % req_cnt)
                #     np.save(save_fn, feature_unpack)

                fp_resp.write('%d. %s\n' % (req_cnt, reqs[i]["data"]["uri"]))
                fp_resp.write(str(res) + '\n')

                #---- new response format (face-feature-v4)
                stream = res["body"]

                fp_feat.write('%d. %s\n' % (req_cnt, reqs[i]["data"]["uri"]))
                # fp_feat.write(str(stream) + '\n')
                fp_feat.write(stream + '\n')

                feature_unpack = np.array(
                    unpack_feature_from_stream(stream), np.float32)
                print '---> unpacked feature from stream: ', feature_unpack

                fp_feat2.write('{}\t{}\t{}\n'.format(reqs[i]["data"]["uri"],
                                                        reqs[i]["data"]["attribute"]["pts"],
                                                        feature_unpack.tolist()))

                img_fn = osp.split(res["result"]["uri"])[-1]
                save_fn = osp.join(save_dir, 'feat_%s.npy' % img_fn)
                np.save(save_fn, feature_unpack)

    fp_resp.close()
    fp_feat.close()
    fp_feat2.close()

    feature_files = []
    file_list = os.listdir(save_dir)
    for it in file_list:
        if it.endswith('.npy') or it.endswith('.bin'):
            feature_files.append(it)

    n_feats = len(feature_files)
    for i in range(n_feats - 1):
        for j in range(i, n_feats):
            compare_feats(
                osp.join(save_dir, feature_files[i]),
                osp.join(save_dir, feature_files[j])
            )
