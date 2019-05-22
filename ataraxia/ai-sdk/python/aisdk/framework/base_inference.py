import json
import zmq.green as zmq
import schema
import time
import os

from aisdk.common.error import ErrorBase, ErrorCV2ImageRead
from aisdk.common.logger import xl
import aisdk.proto as pb
from . import const


class InferenceReq(object):
    def __init__(self):
        ctx = zmq.Context()
        self.inference_req = ctx.socket(zmq.REQ)
        self.inference_req.connect(const.INFERENCE_FORWARD_IN)
        self.monitor_push = ctx.socket(zmq.PUSH)
        self.monitor_push.connect(const.MONIROT_ZMQ_ADDR)
        self.pid = os.getpid()

    def inference_msgs(self, msgs):  # pylint: disable=no-self-use
        assert isinstance(msgs, pb.ForwardMsgs)
        start = time.time()
        self.inference_req.send(msgs.SerializeToString())
        buf = self.inference_req.recv()
        self.monitor_push.send(
            pb.MonitorMetric(
                kind="forward_time",
                pid=str(self.pid),
                value=time.time() - start).SerializeToString())
        msgs_out = pb.ForwardMsgs()
        msgs_out.ParseFromString(buf)
        assert isinstance(msgs_out, pb.ForwardMsgs)
        return msgs_out

    def inference_msg(self, msg):
        assert isinstance(msg, pb.ForwardMsg)
        r = self.inference_msgs(pb.ForwardMsgs(msgs=[msg])).msgs[0]
        assert isinstance(r, pb.ForwardMsg)
        return r


class BaseInferenceServer(object):
    def __init__(self, app_name):
        self.app_name = app_name
        ctx = zmq.Context()
        self.monitor_push = ctx.socket(zmq.PUSH)
        self.monitor_push.connect(const.MONIROT_ZMQ_ADDR)
        self.pid = os.getpid()

    def net_inference(self, request):  # pylint: disable=no-self-use
        assert isinstance(request, pb.InferenceRequest)
        return pb.InferenceResponse(
            code=200, result=json.dumps({
                'hello': 'world'
            }))

    def net_inference_wrap(self, request):
        start = time.time()
        try:
            response = self.net_inference(request)
        except ErrorCV2ImageRead as err:
            xl.warn("net_inference", extra={'reqid': request.reqid})
            response = pb.InferenceResponse(code=err.code, message=err.message)
        except ErrorBase as err:
            xl.warn(
                "net_inference", exc_info=err, extra={'reqid': request.reqid})
            response = pb.InferenceResponse(code=err.code, message=err.message)
        except schema.SchemaError as err:
            xl.warn(
                "net_inference", exc_info=err, extra={'reqid': request.reqid})
            response = pb.InferenceResponse(
                code=400, message='bad api param: {}'.format(err))
        except Exception as err:  # pylint: disable=broad-except
            xl.exception(
                "net_inference", exc_info=err, extra={'reqid': request.reqid})
            response = pb.InferenceResponse(
                code=599,
                message='app {} net_inference unknow error: {}'.format(
                    self.app_name, err))
        self.monitor_push.send(
            pb.MonitorMetric(
                kind="eval_time",
                pid=str(self.pid),
                code=str(response.code),
                value=time.time() - start).SerializeToString())
        return response

    def serve(self):
        network_in_context = zmq.Context()
        network_in = network_in_context.socket(zmq.REP)
        network_in.connect(const.INFERENCE_ZMQ_IN)
        self.monitor_push.send(
            pb.MonitorMetric(
                kind="inference_started_success",
                pid=str(self.pid),
            ).SerializeToString())
        while True:
            buf = network_in.recv()
            request = pb.InferenceRequest()
            request.ParseFromString(buf)
            response = self.net_inference_wrap(request)
            network_in.send(response.SerializeToString())
