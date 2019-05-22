#coding=utf8
from collections import namedtuple
import mxnet as mx
import numpy as np
import six.moves.cPickle as cPickle  # pylint: disable=no-name-in-module,import-error
import time

from aisdk.framework.base_forward import BaseForwardServer

from aisdk.common.mxnet_base import net
from aisdk.common.logger import log
import aisdk.proto as pb
import aisdk.common.mxnet_utils
from . import const

Batch = namedtuple('Batch', ['data'])


class ForwardServer(BaseForwardServer):
    def __init__(self, app_name, batch_size, lock, cfg):
        super(ForwardServer, self).__init__(app_name, batch_size)
        self.lock = lock

        conf = net.NetConfig()
        conf.parse(cfg)
        params_file, sym_file = (conf.file_model, conf.file_symbol)
        ctx = mx.gpu() if conf.use_device == 'GPU' else mx.cpu()

        sym, arg_params, aux_params = aisdk.common.mxnet_utils.load_checkpoint(
            sym_file, params_file)
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

        # pylint: disable=protected-access
        mod.bind(
            for_training=False,
            data_shapes=[('data', (conf.batch_size, 3, conf.image_width,
                                   conf.image_height))],
            label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True)
        self.batch_size = conf.batch_size
        self.width = conf.image_width
        self.height = conf.image_height
        self.mod = mod
        self.mod.forward(
            Batch([
                mx.nd.array(
                    np.zeros((self.batch_size, 3, self.width, self.height)))
            ]))
        self.mod.get_outputs()[0].asnumpy()
        # mxnet 1.3 开始，mod.bind mod.set_params 操作执行是lazy的，会造成收到的第一个请求处理很慢，先预推理一次

    def net_inference(self, msgs):
        assert isinstance(msgs, pb.ForwardMsgs)
        start = time.time()
        img_batch = mx.nd.array(
            np.zeros((self.batch_size, 3, self.width, self.height)))
        for index, msg in enumerate(msgs.msgs):
            arr = np.frombuffer(
                msg.network_input_buf, dtype=np.float32).reshape(
                    (3, self.width, self.height))
            img_batch[index] = arr
        start_forward = time.time()

        with self.lock:
            self.mod.forward(Batch([img_batch]))
            output_batch = self.mod.get_outputs()[0].asnumpy()
        end_forward = time.time()
        msgs_out = []
        for i in range(len(msgs.msgs)):
            msgs_out.append(
                pb.ForwardMsg(
                    network_output_buf=cPickle.dumps(
                        output_batch[i], protocol=cPickle.HIGHEST_PROTOCOL)))
        log.info('{} use time {}, forward time {}, batch_size: {}/{}'.format(
            self.app_name,
            time.time() - start, end_forward - start_forward, len(msgs.msgs),
            self.batch_size))
        return pb.ForwardMsgs(msgs=msgs_out)


def serve(lock):
    srv = ForwardServer(const.app_name, const.cfg['batch_size'], lock,
                        const.cfg)
    srv.serve()


if __name__ == '__main__':
    const.flavor.run_forward(serve)
