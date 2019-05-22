# coding: utf-8
from __future__ import absolute_import, division, generators, nested_scopes, print_function, unicode_literals, with_statement

import os
import sys
import json
from deepdiff import DeepDiff
import multiprocessing
import time
import subprocess
import prometheus_client
from random import shuffle

from aisdk.common.logger import log
import aisdk.proto as pb
from aisdk.common.download_model import parse_tsv_key, read_test_image_with_cache

from aisdk.app.pulp_filter.inference import InferenceServer
import aisdk.app.pulp_filter.forward as inference
from . import const
tsv_path = 'serving/pulp-filter/20181113/set1/reg1113-2.tsv'
tsv_prefix = 'serving/pulp/set1/'


def start_inference():
    lock = multiprocessing.Lock()
    inference.serve(lock)


def run_test_tsv(times=1):
    cases = parse_tsv_key(tsv_path)
    srv = InferenceServer(const.app_name, const.cfg)
    run_test_empty_image(srv)
    run_test_bad_image(srv)
    # run_test_bad_arg(srv)
    for _ in range(times):
        shuffle(cases)
        for index, case in enumerate(cases):
            start = time.time()
            log.info('run case {}/{} image:{}'.format(index + 1, len(cases),
                                                      case[0]))
            request = pb.InferenceRequest(
                data=pb.InferenceRequest.RequestData(
                    body=read_test_image_with_cache(tsv_prefix + case[0]))
                # params=json.dumps({
                #     'limit': 3
                # }
            )
            response = srv.net_inference_wrap(request)
            assert isinstance(response, pb.InferenceResponse)
            assert response.result
            assert response.code == 200
            result = json.loads(response.result)
            confidences = result['confidences'][:1]
            # pulp-filter 后面两个类别 score 是随机的
            confidences.sort(key=lambda x: x['index'])
            actual = json.loads(json.dumps(confidences))
            expected = json.loads(case[1])
            assert DeepDiff(expected, actual, significant_digits=3) == {}
            log.info('use time {}'.format(time.time() - start))


def run_test_empty_image(srv):
    request = pb.InferenceRequest(
        data=pb.InferenceRequest.RequestData(body=b''),
        params=json.dumps({
            'limit': 3
        }))
    response = srv.net_inference_wrap(request)
    assert response.code == 400
    assert response.message == 'cv2 load image from body failed'


def run_test_bad_image(srv):
    request = pb.InferenceRequest(
        data=pb.InferenceRequest.RequestData(body=b'xxxx'),
        params=json.dumps({
            'limit': 3
        }))
    response = srv.net_inference_wrap(request)
    assert response.code == 400
    assert response.message == 'cv2 load image from body failed'


# def run_test_bad_arg(srv):
#     request = pb.InferenceRequest(
#         data=pb.InferenceRequest.RequestData(
#             body=read_test_image_with_cache(
#                 'serving/pulp_filter/set1/Image-tupu-2016-09-01-00-00-327.jpg')),
#         params=json.dumps({
#             'limitxxx': 3
#         }))
#     response = srv.net_inference_wrap(request)
#     assert response.code == 400
#     assert 'Wrong keys' in response.message

#     request = pb.InferenceRequest(
#         data=pb.InferenceRequest.RequestData(
#             body=read_test_image_with_cache(
#                 'serving/pulp_filter/set1/Image-tupu-2016-09-01-00-00-327.jpg')),
#         params=json.dumps({
#             'limit': 100
#         }))
#     response = srv.net_inference_wrap(request)
#     assert response.code == 400
#     assert 'limit too large' in response.message


def test_integration():
    os.chdir('/src')
    prometheus_client.start_http_server(8611)
    child = subprocess.Popen(['/src/res_build/eval_core', 'mq'],
                             stdout=sys.stdout,
                             stderr=sys.stderr)
    p = multiprocessing.Process(target=start_inference)
    p.start()
    try:
        run_test_tsv()
    finally:
        p.terminate()
        child.terminate()
