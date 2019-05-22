#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import logging
import math
import yaml
from easydict import EasyDict as edict
from multiprocessing import cpu_count
import mxnet as mx

from utils import merge_dict

DEFAULT_CONFIG = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), '../default.yaml')


# pylint: disable=E1101
def main(args):
    '''
    Usage:
        train.py        <cfg>

    Arguments:
        <cfg>           path to config file

    Options:
        -h --help              show this help screen
        -v --version           show current version
    '''
    with open(DEFAULT_CONFIG, 'r') as f:
        cfg = edict(yaml.load(f.read()))
    cfg.TRAIN.GPU_IDX = mx.test_utils.list_gpus()
    cfg.TRAIN.ITER.preprocess_threads = cpu_count()
    with open(args['<cfg>'], 'r') as f:
        merge_dict(cfg, edict(yaml.load(f.read())))

    print(cfg)

    log_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger()
    fhandler = logging.FileHandler(cfg.TRAIN.LOG_PATH, mode='w')
    fhandler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fhandler)

    batch_size = cfg.TRAIN.BATCH_SIZE * (len(cfg.TRAIN.GPU_IDX) if len(cfg.TRAIN.GPU_IDX) > 0 else 1)

    iter_args = edict({})
    iter_args.data_name = 'data'
    iter_args.label_name = 'softmax_label'
    iter_args.data_shape = cfg.INPUT_SHAPE
    iter_args.batch_size = batch_size
    iter_args.mean_r, iter_args.mean_g, iter_args.mean_b = cfg.MEAN_RGB[:]
    iter_args.std_r, iter_args.std_g, iter_args.std_b = cfg.STD_RGB[:]
    merge_dict(iter_args, cfg.TRAIN.ITER if cfg.TRAIN.has_key("ITER") else edict({}))

    train_iter_args = edict({})
    merge_dict(train_iter_args, cfg.TRAIN.TRAIN_ITER if cfg.TRAIN.has_key("TRAIN_ITER") else edict({}))
    merge_dict(train_iter_args, iter_args)

    dev_iter_args = edict({})
    merge_dict(dev_iter_args, cfg.TRAIN.DEV_ITER if cfg.TRAIN.has_key("DEV_ITER") else edict({}))
    merge_dict(dev_iter_args, iter_args)

    train_iter = mx.io.ImageRecordIter(**train_iter_args)
    dev_iter = mx.io.ImageRecordIter(**dev_iter_args)

    symbol, arg_params, aux_params = mx.model.load_checkpoint(cfg.TRAIN.FINETUNE.PRETRAINED_MODEL_PREFIX,
                                                              cfg.TRAIN.FINETUNE.PRETRAINED_MODEL_EPOCH)
    symbol = symbol.get_internals()[cfg.TRAIN.FINETUNE.FINETUNE_LAYER + '_output']
    symbol = mx.symbol.FullyConnected(data=symbol,
                                      num_hidden=cfg.NUM_CLASSES,
                                      name='fc-' + str(cfg.NUM_CLASSES))
    # old version of SoftmaxOutput
    # net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    # new version of SoftmaxOutput, mxnet>=1.5.0
    symbol = mx.symbol.SoftmaxOutput(data=symbol,
                                     name='softmax',
                                     grad_scale=1,
                                     ignore_label=-1,
                                     multi_output=0,
                                     use_ignore=0,
                                     preserve_shape=0,
                                     normalization='null',
                                     out_grad=0,
                                     smooth_alpha=cfg.TRAIN.SOFTMAX_SMOOTH_ALPHA)
    arg_params = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})

    kv = mx.kvstore.create('device')
    if not os.path.isdir(os.path.dirname(cfg.TRAIN.OUTPUT_MODEL_PREFIX)):
        os.makedirs(os.path.dirname(cfg.TRAIN.OUTPUT_MODEL_PREFIX))
    checkpoint = mx.callback.do_checkpoint(cfg.TRAIN.OUTPUT_MODEL_PREFIX
                                           if kv.rank == 0
                                           else "%s-%d" % (cfg.TRAIN.OUTPUT_MODEL_PREFIX, kv.rank))

    devices = [mx.gpu(x) for x in cfg.TRAIN.GPU_IDX] if len(cfg.TRAIN.GPU_IDX) > 0 else mx.cpu()
    label_names = ['softmax_label']
    # mode == 'COSINE_DECAY':    # mxnet_version >= 1.4.0
    epoch_size = int(math.ceil(cfg.TRAIN.NUM_SAMPLES / batch_size))     # on all gpus
    max_update = epoch_size * cfg.TRAIN.MAX_EPOCHS
    warmup_steps = epoch_size * cfg.TRAIN.WARMUP_EPOCHS
    lr_scheduler = mx.lr_scheduler.CosineScheduler(max_update, base_lr=cfg.TRAIN.BASE_LR,
                                                   warmup_steps=warmup_steps)

    def _display_lr(BatchEndParam):
        '''
        call back learning rate info
        '''
        if BatchEndParam.nbatch != 0 and BatchEndParam.nbatch % cfg.TRAIN.LOG_INTERVAL == 0:
            logging.info("Epoch[{}] Batch [{}]\tlearning-rate={}".format(
                BatchEndParam.epoch, BatchEndParam.nbatch, BatchEndParam.locals["self"]._optimizer._get_lr(0)))

    batch_end_callbacks = [
        _display_lr,
        mx.callback.Speedometer(batch_size, cfg.TRAIN.LOG_INTERVAL),
    ]

    metrics = mx.metric.CompositeEvalMetric(metrics=[mx.metric.Accuracy(), mx.metric.CrossEntropy()])

    mod = mx.mod.Module(symbol=symbol, context=devices, label_names=label_names)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    # replace all parameters except for the last fully-connected layer with pre-trained model
    mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    mod.fit(train_iter,
            dev_iter,
            eval_metric=metrics,
            begin_epoch=0,
            num_epoch=cfg.TRAIN.MAX_EPOCHS,
            kvstore=kv,
            optimizer='sgd',
            optimizer_params={'learning_rate': cfg.TRAIN.BASE_LR,
                              'momentum': cfg.TRAIN.MOMENTUM,
                              'wd': cfg.TRAIN.WEIGHT_DECAY,
                              'lr_scheduler': lr_scheduler},
            batch_end_callback=batch_end_callbacks,
            epoch_end_callback=checkpoint,
            allow_missing=True)
    score_dev = mod.score(dev_iter, metrics)
    logger.info("Final evaluation on dev-set:")
    for tup in score_dev:
        logger.info("Validation-{}={:.6f}".format(tup[0], tup[1]))


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(main.__doc__, version='0.0.1'))
