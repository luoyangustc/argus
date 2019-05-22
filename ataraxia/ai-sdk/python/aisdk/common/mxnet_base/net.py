# coding: utf-8
from collections import namedtuple
import csv
import os

from aisdk.common.check import value_check, file_check
from aisdk.common.other import parse_crop_size


# pylint: disable=too-many-instance-attributes,too-few-public-methods,invalid-name
class NetConfig(object):
    '''
        Net Config
    '''

    def __init__(self):
        self.file_symbol = None  # deploy.symbol.json
        self.file_model = None  # weight.params
        self.file_synset = None  # labels.csv
        self.file_mean = None  # mean.binaryproto
        self.value_mean = None  #[mean_r,mean_g,mean_b]
        self.value_std = None  #[std_r,std_g,std_b]

        self.file_taglist = None  # custom_files:taglist_file

        self.image_width = None
        self.image_height = None
        self.use_device = None
        self.batch_size = None
        self.model_params = None

        self.custom_files = None
        self.custom_values = None

    def parse(self, configs):
        '''
            parse config input
        '''
        tar_files = value_check(configs, 'model_files')
        self.file_symbol = str(
            file_check(tar_files, 'deploy.symbol.json', False))
        self.file_model = str(file_check(tar_files, 'weight.params'))
        self.file_synset = file_check(tar_files, 'labels.csv', False)
        self.file_mean = file_check(tar_files, 'mean.csv', False)

        self.model_params = configs.get('model_params', {})
        self.custom_files = configs.get('custom_files', {})
        self.file_taglist = file_check(self.custom_files, 'taglist_file',
                                       False)

        self.use_device = value_check(configs, 'use_device', False, "CPU")
        self.batch_size = value_check(configs, 'batch_size', False, 1)

        self.custom_values = configs.get('custom_params', {})
        self.value_mean = value_check(self.custom_values, 'mean_value', False)
        self.value_std = value_check(self.custom_values, 'mean_std', False)

        self.image_width, self.image_height = parse_crop_size( \
                            configs, \
                            model_params=self.model_params, \
                            custom_values=self.custom_values)


_inferConfig = namedtuple('InterConfig',
                          ['deploy_sym', 'labels', 'weight', 'mean', 'custom'])


def parse_infer_config(tar):
    '''
    parse the inference config and check the file's existence
    :return: tuple containing two elements,
    the first config data, the second is error
    '''

    def v(k, e):
        if k not in tar:
            return None, e

        r = tar[k]
        if not os.path.exists(r):
            return None, e

        return r, None

    deploy_sym, err = v('deploy.symbol.json', 'no deploy symbol file')
    if err:
        return None, err

    labels, err = v('labels.csv', 'no label file')
    if err:
        return None, err

    weight, err = v('weight.params', 'no weight file')
    if err:
        return None, err

    mean, _ = v('mean.csv', '')

    custom, _ = v('custom_values', {})

    return _inferConfig(deploy_sym, labels, weight, mean, custom), None


def load_labels(label_file):
    with open(label_file, 'rb') as f:
        return tuple(csv.reader(f, delimiter=','))
