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

from aisdk.app.terror_mixup.inference import InferenceServer
import aisdk.app.terror_mixup.forward as inference

from aisdk.common.download_model import parse_tsv_key, read_test_image_with_cache
from . import const

tsv_path = 'serving/terror-mixup/terror-mixup-201811211548/set20181108/201811211548.tsv'
tsv_prefix = 'serving/terror-mixup/set20181108/'


def start_inference():
    lock = multiprocessing.Lock()
    inference.serve(lock)


def run_test_tsv(times=1):
    cases = parse_tsv_key(tsv_path)
    srv = InferenceServer(const.app_name, const.cfg)
    run_test_empty_image(srv)
    run_test_bad_image(srv)
    for _ in range(times):
        shuffle(cases)
        for index, case in enumerate(cases):
            start = time.time()
            log.info('run case {}/{} image:{}'.format(index + 1, len(cases),
                                                      case[0]))
            request = pb.InferenceRequest(
                data=pb.InferenceRequest.RequestData(
                    body=read_test_image_with_cache(tsv_prefix + case[0])))
            response = srv.net_inference_wrap(request)
            assert isinstance(response, pb.InferenceResponse)
            assert response.result
            assert response.code == 200
            actual = json.loads(response.result)
            expected = json.loads(case[1])
            assert DeepDiff(expected, actual, significant_digits=3) == {}
            log.info('use time {}'.format(time.time() - start))


def run_test_empty_image(srv):
    request = pb.InferenceRequest(
        data=pb.InferenceRequest.RequestData(body=b''))
    response = srv.net_inference_wrap(request)
    assert response.code == 400
    assert response.message == 'cv2 load image from body failed'


def run_test_bad_image(srv):
    request = pb.InferenceRequest(
        data=pb.InferenceRequest.RequestData(body=b'xxxx'))
    response = srv.net_inference_wrap(request)
    assert response.code == 400
    assert response.message == 'cv2 load image from body failed'


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


def test_change_deploy():
    deploy_file = str(const.cfg['model_files']["fine_deploy.prototxt"])
    inference.change_deploy(deploy_file, 16)
