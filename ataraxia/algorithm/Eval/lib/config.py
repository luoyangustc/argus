# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# 
# Modified by Northrend@github.com
# 2018-05-24 10:39:18
# 

'''
Config System.

This file specifies default config options for image classification task on
mxnet, both of mxnet api and gluon api. You should not change values in this 
file. Instead, you should write a config file (in yaml) and use 
merge_cfg_from_file(yaml_file) to load it and override the default options.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from future.utils import iteritems
from past.builtins import basestring
import copy
import logging
import numpy as np
import os
import yaml

class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        '''Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        '''
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]


logger = logging.getLogger(__name__)

__C = AttrDict()
# Consumers can get config by:
#   from lib.config import cfg 
cfg = __C


# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Dummy options
# ---------------------------------------------------------------------------- #
__C.DUMMY = AttrDict()

__C.DUMMY.ENTRYPOINT= "Configuration Entrypoint"

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

__C.TRAIN.LOG_PATH = ""
__C.TRAIN.LOG_LEVEL = "INFO"  # INFO, DEBUG, WARNING, ERROR, CRITICAL
__C.TRAIN.LOG_MODE = "w"  

# ---- training hyper params ----
__C.TRAIN.OUTPUT_MODEL_PREFIX = ""
__C.TRAIN.SAVE_INTERVAL = 1 # save model every n epochs
__C.TRAIN.KV_STORE = b"device" 
__C.TRAIN.USE_GPU = True 
__C.TRAIN.GPU_IDX = [0]
__C.TRAIN.NUM_SAMPLES = 10000 
__C.TRAIN.MAX_EPOCHS = 10
__C.TRAIN.BATCH_SIZE = 4 
__C.TRAIN.METRICS = ["acc"]
__C.TRAIN.METRICS_TOP_K_ACC = 1
__C.TRAIN.PRE_EVALUATION = False 
__C.TRAIN.LOG_INTERVAL = 40  
__C.TRAIN.LOG_NET_PARAMS = False 
__C.TRAIN.LOG_LR = False 
__C.TRAIN.TEST_IO_MODE = False 

# ---- network hyper params ----
__C.TRAIN.XAVIER_INIT = True

__C.TRAIN.FINETUNE = False 
__C.TRAIN.FT = AttrDict()
__C.TRAIN.FT.PRETRAINED_MODEL_PREFIX = ""
__C.TRAIN.FT.PRETRAINED_MODEL_EPOCH = 0
__C.TRAIN.FT.FINETUNE_LAYER = "flatten0"
__C.TRAIN.FT.FREEZE_WEIGHTS = False
__C.TRAIN.FT.LOAD_GLUON_MODEL = False

__C.TRAIN.SCRATCH = False 
__C.TRAIN.SCR = AttrDict()
__C.TRAIN.SCR.NETWORK = "resnet" 
__C.TRAIN.SCR.NUM_LAYERS = 50 
__C.TRAIN.SCR.X_NUM_GROUPS = 32 
__C.TRAIN.SCR.DROP_OUT = 0 

__C.TRAIN.RESUME = False 
__C.TRAIN.RES = AttrDict()
__C.TRAIN.RES.MODEL_PREFIX = ""
__C.TRAIN.RES.MODEL_EPOCH = 0

# ---- data hyper params ----
__C.TRAIN.USE_REC = True 
__C.TRAIN.USE_DALI = False 
__C.TRAIN.TRAIN_REC = "" 
__C.TRAIN.DEV_REC = "" 
__C.TRAIN.TRAIN_LST = "" 
__C.TRAIN.DEV_LST = "" 
__C.TRAIN.TRAIN_IMG_PREFIX = ""
__C.TRAIN.DEV_IMG_PREFIX = ""
__C.TRAIN.PROCESS_THREAD = 4 
__C.TRAIN.INPUT_SHAPE = (3, 224, 224)
__C.TRAIN.RESIZE_RANGE = (800, 1600)
__C.TRAIN.RESIZE_SHAPE = (-1, -1)
__C.TRAIN.RESIZE_SCALE = (1, 1)
__C.TRAIN.RANDOM_RESIZED_CROP = False
__C.TRAIN.RESIZE_AREA = (1, 1)  # 0.08, 1
__C.TRAIN.MEAN_RGB = [123.68, 116.779, 103.939] 
__C.TRAIN.STD_RGB = [58.395, 57.12, 57.375] 
__C.TRAIN.MAX_ROTATE_ANGLE = 0      # 10 
__C.TRAIN.MAX_ASPECT_RATIO = 0.0    # 4.0/3 
__C.TRAIN.MIN_ASPECT_RATIO = 0.0    # 3.0/4 
__C.TRAIN.MAX_SHEAR_RATIO = 0.0 
__C.TRAIN.MAX_IMG_SIZE = 100000.0 
__C.TRAIN.MIN_IMG_SIZE = 0.0 
__C.TRAIN.BRIGHTNESS_JITTER = 0.0
__C.TRAIN.CONTRAST_JITTER = 0.0
__C.TRAIN.SATURATION_JITTER = 0.0
__C.TRAIN.HUE_JITTER = 0.0
__C.TRAIN.PCA_NOISE = 0.0
__C.TRAIN.RANDOM_H = 0 
__C.TRAIN.RANDOM_S = 0 
__C.TRAIN.RANDOM_L = 0 
__C.TRAIN.INTERPOLATION_METHOD = 2
__C.TRAIN.LABEL_WIDTH = 1
__C.TRAIN.DATA_TYPE = None 
__C.TRAIN.SHUFFLE = True 
__C.TRAIN.RAND_CROP = True
__C.TRAIN.RAND_MIRROR = True 
__C.TRAIN.LAST_BATCH_HANDLE = "pad" 

# ---- optimizer hyper params ----
__C.TRAIN.OPTIMIZER = b"sgd"
__C.TRAIN.LR_DECAY_MODE = b"STEP_DECAY"
__C.TRAIN.BASE_LR = 0.1
__C.TRAIN.LR_FACTOR = 0.1
__C.TRAIN.STEP_EPOCHS = list() 
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WARMUP_EPOCHS = 0
__C.TRAIN.WARMUP_BEGIN_LR = 0
__C.TRAIN.WARMUP_MODE = 'linear' 

# ---- classfier hyper params ----
__C.TRAIN.NUM_CLASSES = 1000 
__C.TRAIN.USE_SOFTMAX = True 
__C.TRAIN.SOFTMAX_SMOOTH_ALPHA = 0.0 
__C.TRAIN.USE_SVM = False 
__C.TRAIN.SVM_LOSS = "l2" 
__C.TRAIN.SVM_REG_COEFF = 1.0 

# ---- alternately training setting ----
__C.TRAIN.ALTERNATE = False 
__C.TRAIN.ALT = AttrDict()
__C.TRAIN.ALT.TRAIN_RECS = list()
__C.TRAIN.ALT.DEV_RECS = list() 
__C.TRAIN.ALT.NUM_NETS = 2 
__C.TRAIN.ALT.NUM_CLASSES = list() 
__C.TRAIN.ALT.NUM_SAMPLES = list() 
__C.TRAIN.ALT.PHASE = 1
__C.TRAIN.ALT.WEIGHT_DECAYS = list()
__C.TRAIN.ALT.MOMENTUMS = list() 

# ---- precision-guided training setting ----  
__C.TRAIN.PRECISION_GUIDED = False
__C.TRAIN.PG = AttrDict()
__C.TRAIN.PG.LOG_GEN_GUIDER = False 
__C.TRAIN.PG.INPUT_IMG_LST = ""
__C.TRAIN.PG.INPUT_IMG_PATH = ""
__C.TRAIN.PG.IMG_CACHE_PATH = ""
__C.TRAIN.PG.IMG_CACHE_LST = ""
__C.TRAIN.PG.IMG_CACHE_PREFIX = "PG_"
__C.TRAIN.PG.IMG_CACHE_EXT = ".jpg"
__C.TRAIN.PG.GIUDING_MODEL_PREFIX = ""
__C.TRAIN.PG.GIUDING_MODEL_EPOCH = 0
__C.TRAIN.PG.OUTPUT_GROUP = list() 
__C.TRAIN.PG.CLASSIFIER_WEIGHTS = "" 
__C.TRAIN.PG.TARGET_SHAPE = (8, 8)  # after down-sapmling 


# ---------------------------------------------------------------------------- #
# Tesing options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

__C.TEST.LOG_PATH = ""
__C.TEST.LOG_LEVEL = "INFO"  # INFO, DEBUG, WARNING, ERROR, CRITICAL
__C.TEST.LOG_MODE = "w"  

# ---- test setting ----
__C.TEST.INPUT_IMG_LST = ""
__C.TEST.INPUT_IMG_PREFIX = ""
__C.TEST.INPUT_CAT_FILE = ""
__C.TEST.OUTPUT_JSON_PATH = ""
__C.TEST.MODEL_PREFIX = ""
__C.TEST.MODEL_EPOCH = 0
__C.TEST.KV_STORE = b"device" 
__C.TEST.USE_GPU = True 
__C.TEST.GPU_IDX = [0]     # only single gpu supported for now
__C.TEST.PROCESS_NUM = 1 
__C.TEST.MUTABLE_IMAGES_TEST = False
__C.TEST.BATCH_SIZE = 1 
__C.TEST.INPUT_SHAPE = (3, 224, 224)
__C.TEST.RESIZE_KEEP_ASPECT_RATIO = False
__C.TEST.RESIZE_WH = (224, 224) 
__C.TEST.RESIZE_MIN_MAX = (256, 0) 
__C.TEST.MEAN_RGB = [123.68, 116.779, 103.939] 
__C.TEST.STD_RGB = [58.395, 57.12, 57.375] 
__C.TEST.CENTER_CROP = False
__C.TEST.MULTI_CROP = False
__C.TEST.MULTI_CROP_NUM = 3
__C.TEST.HORIZENTAL_FLIP = False
__C.TEST.USE_BASENAME = True
__C.TEST.FNAME_PARENT_LEVEL = 1 
__C.TEST.TOP_K = 1 
__C.TEST.LOG_ALL_CONFIDENCE = True
__C.TEST.CAT_NAME_POS = 1
__C.TEST.CAT_FILE_SPLIT = " "

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set(
    {
        'TRAIN.DUMMY_1',
        'TEST.DUMMY_2',
    }
)

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'MRCNN.MASK_HEAD_NAME': 'MRCNN.ROI_MASK_HEAD',
    'TRAIN.DATASET': (
        'TRAIN.DATASETS',
        "Also convert to a tuple, e.g., " +
        "'coco_2014_train' -> ('coco_2014_train',) or " +
        "'coco_2014_train:coco_2014_valminusminival' -> " +
        "('coco_2014_train', 'coco_2014_valminusminival')"
    ),
}


# ---------------------------------------------------------------------------- #
# Renamed modules
# If a module containing a data structure used in the config (e.g. AttrDict)
# is renamed/moved and you don't want to break loading of existing yaml configs
# (e.g. from weights files) you can specify the renamed module below.
# ---------------------------------------------------------------------------- #
_RENAMED_MODULES = {
    'utils.collections': 'detectron.utils.collections',
}


def assert_and_infer_cfg(cache_urls=True, make_immutable=True):
    '''Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    '''
    if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
        __C.RPN.RPN_ON = True
    if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
        __C.TEST.PRECOMPUTED_PROPOSALS = False
    if cache_urls:
        cache_cfg_urls()
    if make_immutable:
        cfg.immutable(True)


# def cache_cfg_urls():
#     '''Download URLs in the config, cache them locally, and rewrite cfg to make
#     use of the locally cached file.
#     '''
#     __C.TRAIN.WEIGHTS = cache_url(__C.TRAIN.WEIGHTS, __C.DOWNLOAD_CACHE)
#     __C.TEST.WEIGHTS = cache_url(__C.TEST.WEIGHTS, __C.DOWNLOAD_CACHE)
#     __C.TRAIN.PROPOSAL_FILES = tuple(
#         cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TRAIN.PROPOSAL_FILES
#     )
#     __C.TEST.PROPOSAL_FILES = tuple(
#         cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TEST.PROPOSAL_FILES
#     )


# def get_output_dir(datasets, training=True):
#     '''Get the output directory determined by the current global config.'''
#     assert isinstance(datasets, (tuple, list, basestring)), \
#         'datasets argument must be of type tuple, list or string'
#     is_string = isinstance(datasets, basestring)
#     dataset_name = datasets if is_string else ':'.join(datasets)
#     tag = 'train' if training else 'test'
#     # <output-dir>/<train|test>/<dataset-name>/<model-type>/
#     outdir = os.path.join(__C.OUTPUT_DIR, tag, dataset_name, __C.MODEL.TYPE)
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#     return outdir


def load_cfg(cfg_to_load):
    '''
    Wrapper around yaml.load used for maintaining backward compatibility
    '''
    assert isinstance(cfg_to_load, (file, basestring)), \
        'Expected {} or {} got {}'.format(file, basestring, type(cfg_to_load))
    if isinstance(cfg_to_load, file):
        cfg_to_load = ''.join(cfg_to_load.readlines())
    if isinstance(cfg_to_load, basestring):
        for old_module, new_module in iteritems(_RENAMED_MODULES):
            # yaml object encoding: !!python/object/new:<module>.<object>
            old_module, new_module = 'new:' + old_module, 'new:' + new_module
            cfg_to_load = cfg_to_load.replace(old_module, new_module)
    return yaml.load(cfg_to_load)


def merge_cfg_from_file(cfg_filename):
    '''
    Load a yaml config file and merge it into the global config.
    '''
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(load_cfg(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    '''
    Merge `cfg_other` into the global config.
    '''
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    '''
    Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    '''
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    '''
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    '''
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        logger.warn(
            'Deprecated config key (ignoring): {}'.format(full_key)
        )
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
        format(full_key, new_key, msg)
    )


def _decode_cfg_value(v):
    '''Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    '''
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    '''
    Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    '''
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a

if __name__ == '__main__':
    pass
