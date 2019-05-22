#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import time
import json
import cv2
import re
import docopt
import logging
import yaml
from collections import namedtuple
from easydict import EasyDict as edict
import mxnet as mx
import numpy as np

__dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(__dir, "../train"))

from utils import merge_dict

DEFAULT_CONFIG = os.path.join(__dir, '../default.yaml')


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
    with open(args['<cfg>'], 'r') as f:
        merge_dict(cfg, edict(yaml.load(f.read())))

    symbol, arg_params, aux_params = mx.model.load_checkpoint(cfg.TEST.MODEL_PREFIX,
                                                              cfg.TEST.MODEL_EPOCH)
    model = mx.mod.Module(symbol=symbol, context=[mx.gpu(cfg.TEST.GPU_ID)], label_names=None)
    model.bind(for_training=False,
               data_shapes=[('data', tuple([cfg.TEST.BATCH_SIZE] + cfg.INPUT_SHAPE))],
               label_shapes=model._label_shapes)
    model.set_params(arg_params, aux_params, allow_missing=True)

    img_batch = mx.nd.array(np.zeros(tuple([cfg.TEST.BATCH_SIZE] + cfg.INPUT_SHAPE)))
    outputs = []
    filenames = []
    index = 0
    with open(cfg.TEST.IMAGE_LIST_FILE, 'r') as f:
        for line in f:
            file = line.strip().split()[0]

            img = cv2.imread(file)
            if np.shape(img) == tuple():
                print("empty image: {}".format(file))
                continue

            filenames.append(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(float)
            img = cv2.resize(img, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[2]))
            img -= cfg.MEAN_RGB
            img /= cfg.STD_RGB
            # (h,w,c) => (c,h,w)
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img_batch[index] = img
            index += 1

            if index < cfg.TEST.BATCH_SIZE:
                continue

            Batch = namedtuple('Batch', ['data'])
            print("*", end='')
            sys.stdout.flush()
            model.forward(Batch([img_batch]))
            index = 0
            output_prob_batch = model.get_outputs()[0].asnumpy()
            for i in range(cfg.TEST.BATCH_SIZE):
                outputs.append(output_prob_batch[i])

    if index > 0:
        Batch = namedtuple('Batch', ['data'])
        model.forward(Batch([img_batch]))
        output_prob_batch = model.get_outputs()[0].asnumpy()
        for i in range(index):
            outputs.append(output_prob_batch[i])

    results = dict()
    for i, output_prob in enumerate(outputs):
        index_list = output_prob.argsort()
        rate_list = output_prob[index_list]
        _index_list = index_list.tolist()[:][::-1]
        _rate_list = rate_list.tolist()[:][::-1]
        results[filenames[i]] = {
            'Top-1 Index': _index_list,
            'Confidence': [float(x) for x in list(output_prob)],
        }

    with open(cfg.TEST.OUTPUT_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(main.__doc__, version='0.0.1'))
