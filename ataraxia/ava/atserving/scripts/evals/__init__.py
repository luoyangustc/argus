'''
    Eval Scripts
'''

#!/usr/bin/python
# -*- coding: UTF-8 -*-

from utils import metrics
from evals.utils import net_preprocess_handler, CTX
from eval import create_net, net_inference


@net_preprocess_handler
def _net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


try:
    from eval import net_preprocess
except ImportError:
    net_preprocess = _net_preprocess
