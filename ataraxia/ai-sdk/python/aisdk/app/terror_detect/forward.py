import six.moves.cPickle as cPickle  # pylint: disable=no-name-in-module,import-error
import time

import os
import os.path
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

from aisdk.framework.base_forward import BaseForwardServer

from aisdk.common.logger import log
import aisdk.proto as pb
import aisdk.common.mxnet_utils
import aisdk.common.other
from . import const

import mxnet as mx

from rfcn.config.config import config, update_config
# pylint: disable=wildcard-import
from rfcn.symbols import *
from core.tester import Predictor, im_detect


def convert_context(params, ctx):
    """
    :param params: dict of str to NDArray
    :param ctx: the context to convert to
    :return: dict of str of NDArray with context ctx
    """
    new_params = dict()
    for k, v in params.items():
        new_params[k] = v.as_in_context(ctx)
    return new_params


def load_param(arg_params, aux_params, convert=False, ctx=None, process=False):
    if convert:
        if ctx is None:
            ctx = mx.cpu()
        arg_params = convert_context(arg_params, ctx)
        aux_params = convert_context(aux_params, ctx)
    if process:
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
    return arg_params, aux_params


def get_net(cfg, ctx, arg_params, aux_params, has_rpn):
    # pylint: disable=eval-used
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)

    # infer shape
    SHORT_SIDE = config.SCALES[0][0]
    LONG_SIDE = config.SCALES[0][1]
    DATA_NAMES = ['data', 'im_info']
    LABEL_NAMES = None
    DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1,
                                                                         3))]
    LABEL_SHAPES = None
    data_shape_dict = dict(DATA_SHAPES)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(
        arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]),
                                 max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30,
                                        5)))

    # create predictor
    predictor = Predictor(
        sym,
        DATA_NAMES,
        LABEL_NAMES,
        context=ctx,
        max_data_shapes=max_data_shape,
        provide_data=[DATA_SHAPES],
        provide_label=[LABEL_SHAPES],
        arg_params=arg_params,
        aux_params=aux_params)
    return predictor


class ForwardServer(BaseForwardServer):
    def __init__(self, app_name, batch_size, lock, cfg):
        super(ForwardServer, self).__init__(app_name, batch_size)
        self.lock = lock

        use_device = cfg['use_device']
        ctx = [mx.gpu()] if use_device == 'GPU' else [mx.cpu()]
        self.classes = aisdk.common.other.make_synset(
            cfg['model_files']['labels.csv'])

        sym_file = cfg['model_files']['deploy.symbol.json']
        params_file = cfg['model_files']['weight.params']
        _, arg_params, aux_params = aisdk.common.mxnet_utils.load_checkpoint(
            sym_file, params_file)
        arg_params, aux_params = load_param(
            arg_params, aux_params, process=True)

        yaml_file = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'resnet.yaml')
        update_config(yaml_file)

        self.predictor = get_net(config, ctx, arg_params, aux_params, True)

    def net_inference(self, msgs):
        assert isinstance(msgs, pb.ForwardMsgs)
        start = time.time()
        start_forward = time.time()
        assert len(msgs.msgs) == 1
        arr = cPickle.loads(msgs.msgs[0].network_input_buf)
        data = [[mx.nd.array(arr['im_array']), mx.nd.array(arr['im_info'])]]
        data_batch = mx.io.DataBatch(
            data=data,
            label=[None],
            provide_data=arr['data_shapes'],
            provide_label=[None])
        with self.lock:
            # https://github.com/ataraxialab/Deformable-ConvNets/blob/master/rfcn/core/tester.py#L124
            scores, boxes, _ = im_detect(self.predictor, data_batch,
                                         ['data', 'im_info'], arr['im_scale'],
                                         config)
        end_forward = time.time()
        msgs_out = []
        for _ in range(len(msgs.msgs)):
            output = {
                'scores': scores[0],
                'boxes': boxes[0],
            }
            msgs_out.append(
                pb.ForwardMsg(
                    network_output_buf=cPickle.dumps(
                        output, protocol=cPickle.HIGHEST_PROTOCOL)))
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
