# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "./src")

import pyximport

pyximport.install()
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
from config import Config, LabelType
from timer import Timer
import json
import sys
import traceback
import time
from engine import FilterEngine

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *


@create_net_handler
def create_net(configs):
    CTX.logger.info("load configs: %s", configs)
    type_map = {}
    for t in LabelType:
        type_map[t.value] = True

    if "custom_files" in configs and "keyword_file" in configs["custom_files"] and configs["custom_files"]["keyword_file"]:
        CTX.logger.info("load the keyword_file:{}".format(
            configs["custom_files"]["keyword_file"]))
        engine_model = FilterEngine(
            dfa=configs["custom_files"]["keyword_file"], regular=Config.SENSITIVE_RULES_PATH)
    else:
        engine_model = FilterEngine(
            dfa=Config.SENSITIVE_KEYS_PATH, regular=Config.SENSITIVE_RULES_PATH)
    return {"engine": engine_model, "batch_size": configs['batch_size'], "type_check": type_map}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    engine = model['engine']
    type_map = model['type_check']
    CTX.logger.info("inference begin ...")
    resps = []
    index = 0
    try:
        _t1 = time.time()
        for idx, req in enumerate(reqs):
            index = idx

            if "body" not in req["data"]:
                CTX.logger.error("bad request:{}".format(req))
                raise ErrorBase(
                    code=400, msg="body params should be in data")

            texts = json.loads(req["data"]["body"], encoding="utf-8")
            if not isinstance(texts, list):
                CTX.logger.error(
                    "texts should be list but got:{}".format(req))
                raise ErrorBase(
                    code=400, msg="text body parameter should be list")

            type_lst = None
            if "params" in req and "type" in req["params"]:
                type_lst = req["params"]["type"]
                if type_lst:
                    type_lst = list(set(type_lst))
                for typ in type_lst:
                    if typ not in type_map:
                        CTX.logger.error(
                            "type list error with bad request params:{}".format(req))
                        raise ErrorBase(
                            code=400, msg="type list of params is illegal")

            result = engine.check(texts, type_lst)
            resps.append({"code": 0, "message": "", "result": result})

        _t2 = time.time()
        CTX.logger.info("forward: %f", _t2 - _t1)
    except ErrorBase as e:
        return [], e.code, str(e)

    except Exception as e:
        if isinstance(e, ValueError) or isinstance(e, TypeError):
            CTX.logger.error("bad request:{}".format(reqs[index]))
            return [], 400, str(e)
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)

    return resps, 0, ''
