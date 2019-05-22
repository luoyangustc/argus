import os

from zmq.error import Again
import zmq.green as zmq
import aisdk.proto as pb
from aisdk.common.logger import log
from . import const


class BaseForwardServer(object):
    def __init__(self, app_name, batch_size):
        ctx = zmq.Context()
        self.app_name = app_name
        self.batch_size = batch_size
        self.monitor_push = ctx.socket(zmq.PUSH)
        self.monitor_push.connect(const.MONIROT_ZMQ_ADDR)
        self.pid = os.getpid()

    def net_inference(self, msgs):  # pylint: disable=no-self-use
        assert isinstance(msgs, pb.ForwardMsgs)
        msgs_out = []
        for i in msgs:
            msg_out = pb.ForwardMsg()
            msg_out.network_output_buf = msgs[i].network_input_buf
            msgs_out.append(msg_out)
        return pb.ForwardMsgs(msgs=msgs_out)

    def net_inference_wrap(self, msgs):
        assert isinstance(msgs, pb.ForwardMsgs)
        msgs_out = self.net_inference(msgs)
        assert isinstance(msgs_out, pb.ForwardMsgs)
        for i in range(len(msgs_out.msgs)):
            msgs_out.msgs[i].uuid = msgs.msgs[i].uuid
        return msgs_out

    def serve(self):
        max_batch_size = self.batch_size

        log.info('run forward max_batch_size:%s', max_batch_size)
        network_in_context = zmq.Context()
        network_in = network_in_context.socket(zmq.PULL)
        network_in.connect(const.FORWARD_IN)

        network_out_context = zmq.Context()
        network_out = network_out_context.socket(zmq.PUSH)
        network_out.connect(const.FORWARD_OUT)
        inputs = []
        self.monitor_push.send(
            pb.MonitorMetric(
                kind="forward_started_success",
                pid=str(self.pid)).SerializeToString())
        while True:

            def process(buf):
                msg = pb.ForwardMsg()
                msg.ParseFromString(buf)
                inputs.append(msg)

            buf = network_in.recv()
            process(buf)
            while len(inputs) < max_batch_size:
                try:
                    buf = network_in.recv(zmq.NOBLOCK)
                    process(buf)
                except Again:
                    break
            if not inputs:
                continue
            outputs = self.net_inference_wrap(
                pb.ForwardMsgs(msgs=inputs[:max_batch_size]))
            network_out.send(outputs.SerializeToString())
            inputs = inputs[max_batch_size:]
