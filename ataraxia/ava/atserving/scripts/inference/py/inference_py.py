#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import os
import tempfile
import traceback

from inference_pb2 import CreateParams, \
    InferenceRequest, InferenceRequests, InferenceResponses

from evals import \
    create_net as eval_create_net, \
    net_preprocess as eval_net_preprocess, \
    net_inference as eval_net_inference


def create_net(args):

    try:

        params = CreateParams()
        params.ParseFromString(args)

        dir = tempfile.mkdtemp()
        print("data dir: ", dir)

        model_params = {}
        if params.HasField("model_params") and params.model_params is not None:
            model_params = json.loads(params.model_params)
        model_files = {}
        if params.model_files is not None:
            os.makedirs(os.path.join(dir, 'models'))
            for file in params.model_files:
                file_dir = os.path.dirname(file.name)
                if file_dir.strip() != "" and not os.path.exists(os.path.join(dir, 'models', file_dir)):
                    os.makedirs(os.path.join(dir, 'models', file_dir))
                with open(os.path.join(dir, 'models', file.name), 'w+') as f:
                    f.write(file.body)
                model_files[file.name] = os.path.join(dir, 'models', file.name)

        custom_params = {}
        if params.HasField("custom_params") and params.custom_params is not None:
            custom_params = json.loads(params.custom_params)
        custom_files = {}
        if params.custom_files is not None:
            os.makedirs(os.path.join(dir, 'customs'))
            for file in params.custom_files:
                file_dir = os.path.dirname(file.name)
                if file_dir.strip() != "" and not os.path.exists(os.path.join(dir, 'customs', file_dir)):
                    os.makedirs(os.path.join(dir, 'customs', file_dir))
                with open(os.path.join(dir, 'customs', file.name), 'w+') as f:
                    f.write(file.body)
                custom_files[file.name] = os.path.join(dir, 'customs', file.name)

        _args = {
            "app": params.env.app,
            "workspace": params.env.workspace,
            "batch_size": params.batch_size,
            "use_device": params.use_device,
            "model_files": model_files,
            "model_params": model_params,
            "custom_files": custom_files,
            "custom_params": custom_params,
        }

        return eval_create_net(_args)
    except Exception as _e:
        print("create_net: ", _e)
        raise _e


def net_preprocess(ctx, args):

    try:
        request = InferenceRequest()
        request.ParseFromString(args)

        req = {}
        if request.HasField("params") and request.params is not None:
            params = json.loads(request.params)
            req["params"] = params
        if len(request.datas) > 0:
            datas = []
            for data in request.datas:
                attribute = {}
                if data.HasField("attribute") and data.attribute is not None:
                    attribute = json.loads(data.attribute)
                datas = datas + [{"uri": data.uri, "body": data.body, "attribute": attribute}]
            req["data"] = datas
        else:
            attribute = {}
            if request.data.HasField("attribute") and request.data.attribute is not None:
                attribute = json.loads(request.data.attribute)
            req["data"] = {
                "uri": request.data.uri,
                "body": request.data.body,
                "attribute": attribute,
            }

        ret, code, err = eval_net_preprocess(ctx, req, "xx")
        if err is not None and err != '':
            return "", code, err

        response = InferenceRequest()
        if ret.has_key('params') and ret['params'] is not None:
            response.params = json.dumps(ret['params'])
        if type(ret['data']) == list:
            for data in ret['data']:
                _data = response.datas.add()
                if data.has_key('attribute') and data['attribute'] is not None:
                    _data.attribute = json.dumps(data['attribute'])
                _data.uri = data['uri']
                _data.body = data['body']
        else:
            data = ret.get('data', {})
            if data.has_key('attribute') and data['attribute'] is not None:
                response.data.attribute = json.dumps(data['attribute'])
            response.data.uri = data['uri']
            response.data.body = data['body']

        return response.SerializeToString(), 0, ""
    except Exception as _e:
        print("net_preprocess: ", _e)
        raise _e


def net_inference(ctx, args):

    try:
        requests = InferenceRequests()
        requests.ParseFromString(args)

        reqs = []
        for request in requests.requests:
            params = {}
            if request.HasField('params') and request.params is not None:
                params = json.loads(request.params)
            req = {"params": params}
            if len(request.datas) > 0:
                datas = []
                for data in request.datas:
                    attribute = {}
                    if data.HasField('attribute') and data.attribute is not None:
                        attribute = json.loads(data.attribute)
                    datas = datas + [{"uri": data.uri, "body": data.body, "attribute": attribute}]
                req["data"] = datas
            else:
                attribute = {}
                if request.data.HasField('attribute') and request.data.attribute is not None:
                    attribute = json.loads(request.data.attribute)
                req["data"] = {
                    "uri": request.data.uri,
                    "body": request.data.body,
                    "attribute": attribute,
                }
            reqs = reqs + [req]

        ret, code, err = eval_net_inference(ctx, reqs, "xx")
        if err is not None and err != '':
            return "", code, err

        responses = InferenceResponses()
        for resp in ret:
            response = responses.responses.add()
            response.code = resp.get('code', 0)
            response.message = resp.get('message', '')
            response.result = json.dumps(resp.get('result', {}))
            response.body = resp.get('body', '')

        return responses.SerializeToString(), 0, ""
    except Exception as _e:
        print("net_inference: ", _e, traceback.format_exc())
        raise _e
