#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging


def init_logger():

    logger = logging.getLogger("atserving")
    if not logger.propagate:
        return logger
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(reqid)s] [%(levelname)s] %(pathname)s:%(lineno)d: %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

logger = init_logger()

