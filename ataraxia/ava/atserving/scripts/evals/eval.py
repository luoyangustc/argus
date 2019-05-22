#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''for local
'''

from evals.utils import CTX, create_net_handler, net_inference_handler


@create_net_handler
def create_net(cfg):
    '''
        Nothing to do
    '''
    _ = cfg
    CTX.logger.info("%s", cfg)
    return {'data': [1, 2, 3, 4]}, 0, ''


@net_inference_handler
def net_inference(model, args):
    '''
        Nothing to do
    '''
    _, _ = (model, args)
    CTX.logger.info("%s", model.get('data', []))
    return [
        {
            'result': {
                'data': model.get('data', []),
                'length': len(args[0].get('data', {}).get('body', '')),
            },
        },
    ], 0, ""
