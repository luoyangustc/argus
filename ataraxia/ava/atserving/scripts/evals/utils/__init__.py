#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
    utils
'''

import csv
import functools
import json
import logging
import os
import threading
import traceback

from prometheus_client import CollectorRegistry, generate_latest, \
    Histogram

from evals.utils.logger import logger
from evals.utils.error import ErrorConfig, ErrorFileNotExist


def _make_synset(path_csv):
    '''
        generate synset list from synset csv
        csv format:
        <label_index>,"<label_name>"
        ...
        synset_list: [[class_index, class_name],...]
    '''
    synset_list = []
    with open(path_csv, 'r') as file_csv:
        if file_csv:
            read = csv.reader(file_csv)
            synset_list = [r for r in read]
    return synset_list


def _make_tag(path_tag):
    '''
        generate tags dict from taglist file
    '''
    buff = ''
    with open(path_tag, 'r') as ftag:
        for line in ftag:
            buff += line.strip()
    tag = json.loads(buff)
    return tag


def file_check(configs, key, must=True):
    '''
        get file with check
    '''
    target = ""
    if key not in configs:
        if must:
            raise ErrorConfig(key)
        return None
    target = configs[key]
    if not os.path.isfile(target):
        raise ErrorFileNotExist(target)
    return target


def value_check(configs, key, must=True, default=None):
    '''
        get value with check
    '''
    if key not in configs or configs[key] is None:
        if not must:
            return default
        raise ErrorConfig(key)
    return configs[key]


def infer_input_unmarshal(model, args):
    '''
        unmarshal inference input to net, reqid, reqs
    '''
    net = model["net"]
    args = json.loads(args)
    reqid = args["reqid"]
    reqs = args["reqs"]
    return net, reqid, reqs


def infer_output_marshal(ret, headers=None):
    '''
        marshal inference output to json string
    '''
    output = {"results": ret}
    if headers is not None:
        output["headers"] = headers.marshal()
    return json.dumps(output)


def parse_params_file(params_file_name):
    '''
    parse the 'params' file whose filename has
    pattern `prefix_model`-`epoch`.params
    :params params_file_name params filename
    '''
    elems = params_file_name[:-len('.params')].split('-')

    return '-'.join(elems[:-1]), int(elems[-1])


def parse_crop_size(configs,
                    model_params=None, custom_values=None,
                    default_image_width=0, default_image_height=0):
    '''
        RETURN
            image_width, image_height
    '''
    import re

    if model_params is None:
        model_params = value_check(configs, 'model_params', False, {})
    if custom_values is None:
        custom_values = value_check(configs, 'custom_values', False, {})

    crop_size = value_check(model_params, 'cropSize', False, "")
    if crop_size != "":
        groups = re.match('^([0-9]+)(x([0-9]+))?$', crop_size).groups()
        if len(groups) != 3:
            raise ErrorConfig("model_params.cropSize")
        if groups[2] is None:
            return int(groups[0]), int(groups[0])
        else:
            return int(groups[0]), int(groups[2])

    image_width = value_check(configs, 'image_width', False, 0)
    if image_width == 0:
        image_width = value_check(custom_values, 'image_width', False, 0)
    if image_width == 0:
        image_width = default_image_width

    image_height = value_check(configs, 'image_height', False, 0)
    if image_height == 0:
        image_height = value_check(custom_values, 'image_height', False, 0)
    if image_height == 0:
        image_height = default_image_height
    if image_height == 0:
        image_height = image_width

    return image_width, image_height


################################################################################

'''
    .app    app name
    .reqID  request ID
    .logger logger
    .header {"": [], ...}
'''
CTX = threading.local()


def create_net_handler(func):
    '''handle net_init
    Args:
        req:
                {
                    "batch_size": <int>,
                    "use_device": "CPU"|"GPU",
                    "custom_files": {}
                    "custom_params": {}
                    "model_files": {}
                    "model_params": {}
                    "workspace": ""
                    "app": ""
                }
    Returns:
        a dict
        example:
        {
            "features": [],
        }
    Raises:
        None
    '''
    @functools.wraps(func)
    def wrapper(req):
        '''
            handler func
        '''
        CTX.reqID = "net_init"
        CTX.app = req.get('app', '')
        CTX.logger = logging.LoggerAdapter(logger, {'reqid': CTX.reqID})

        try:
            ret, code, err = func(req)
        except Exception as _e:
            CTX.logger.error(_e)
            CTX.logger.error("backtrace: %s", traceback.format_exc())
            raise _e
        return ret, code, err
    return wrapper


def net_preprocess_handler(func):
    '''handle net_preprocess
    Args:
        model:  a Dict, example:
                {
                    'features': []
                }
        req:
                {
                   "data": {
                       "uri": ""
                       "body": 
                   },
                   "params": {}
                }
    Returns:
       {
           "data": {
               "uri": ""
               "body":
           },
           "params": {}
       }
    Raises:
        None
    '''
    @functools.wraps(func)
    def wrapper(model, req, reqid=''):
        '''handle func
        '''
        CTX.reqID = reqid  # TODO
        CTX.logger = logging.LoggerAdapter(logger, {'reqid': CTX.reqID})
        try:
            ret, code, error = func(model, req)
        except Exception as _e:
            CTX.logger.error("exception: %s", _e.message)
            CTX.logger.error("backtrace: %s", traceback.format_exc())
            raise _e
            # return None, 599, _e.message

        return ret, code, error
    return wrapper


def net_inference_handler(func):
    '''handle net_inference
    Args:
        model:  a Dict, example:
                {
                    'features': []
                }
        req:
                [
                    {
                        "data": {
                            "uri": ""
                            "body": 
                        },
                        "params": {}
                    }
                ]
    Returns:
        [
            {
                "code": 0,
                "message": "",
                "header": {},
                "result": {},
                "result_file": ""
            }
        ]
    Raises:
        None
    '''
    @functools.wraps(func)
    def wrapper(model, req, reqid=''):
        '''handle func
        '''
        CTX.reqID = reqid  # TODO
        CTX.logger = logging.LoggerAdapter(logger, {'reqid': CTX.reqID})
        try:
            ret, code, error = func(model, req)
        except Exception as _e:
            CTX.logger.error("exception: %s", _e.message)
            CTX.logger.error("backtrace: %s", traceback.format_exc())
            raise _e
            # return None, 599, _e.message

        def encode_result(result):
            '''encode result
            '''
            header = result.get('header', {})
            if not header.has_key('X-Origin-A'):
                header['X-Origin-A'] = ["{}:1".format(CTX.app.upper())]
            result['header'] = header
            return result

        ret = map(encode_result, ret)
        return ret, code, error
    return wrapper

################################################################################


REGISTRY = CollectorRegistry()


def metrics():
    return generate_latest(REGISTRY)


MONITOR_RESPONSETIME = Histogram(
    'response_time',
    'Response time of requests',
    namespace='ava',
    subsystem='serving_eval',
    labelnames=('method', 'error', 'number'),
    buckets=(0,
             0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
             1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 20, 30, 40, 50, 60,
             ),
    registry=REGISTRY,
)


def monitor_rt_inference(number=1, error='NIL'):
    '''Histogram ResponseTime @ inference
    '''
    return MONITOR_RESPONSETIME.labels('py.inference', error, number)


def monitor_rt_load(number=1, error='NIL'):
    '''Histogram ResponseTime @ load
    '''
    return MONITOR_RESPONSETIME.labels('py.load', error, number)


def monitor_rt_transform(number=1, error='NIL'):
    '''Histogram ResponseTime @ transform
    '''
    return MONITOR_RESPONSETIME.labels('py.transform', error, number)


def monitor_rt_forward(number=1, error='NIL'):
    '''Histogram ResponseTime @ forward
    '''
    return MONITOR_RESPONSETIME.labels('py.forward', error, number)


def monitor_rt_post(number=1, error='NIL'):
    '''Histogram ResponseTime @ post
    '''
    return MONITOR_RESPONSETIME.labels('py.post', error, number)
