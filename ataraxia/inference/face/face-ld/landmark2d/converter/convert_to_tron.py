from __future__ import print_function

from tron.util import mkdir_p
from config.config_customs import *
from config.config_examples import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Convert model to tron!')
    parser.add_argument('--model_root', '-d', required=True, help='The root folder of the models to be converted.')
    parser.add_argument('--config_name', '-c', required=True, help='The suffix name of the model config function.')
    parser.add_argument('--save_root', '-s', default='model_tron', help='The root folder to save the tron model.')
    parser.add_argument('--copy_params', '-p', action='store_true', help='Copy source model weights.')
    parser.add_argument('--merge_op', '-m', action='store_true', help='Merge operators.')
    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    args = parse_args()

    meta_net_info = eval('get_config_' + args.config_name)()

    mkdir_p(args.save_root)

    model_type = meta_net_info['model_type']
    if model_type == 'caffe':
        from tron.caffe2tron import caffe2tron
        tron_net = caffe2tron(args.model_root, meta_net_info, args.copy_params)
    elif model_type == 'mxnet':
        from tron.mxnet2tron import mxnet2tron
        tron_net = mxnet2tron(args.model_root, meta_net_info, args.copy_params)
    else:
        raise ValueError('Currently only support convert caffe or mxnet model!', model_type)

    if args.merge_op:
        from tron.merger import Merge
        merged_net = Merge(tron_net, args.copy_params)

    save_name = args.save_root + '/' + meta_net_info['save_name']

    if args.copy_params:
        save_path = save_name + '.tronmodel'
        tron_net.write_proto_to_binary(save_path)
        print('Convert successful, model has been written to ' + save_path)
        if args.merge_op:
            save_path = save_name + '_merged.tronmodel'
            merged_net.write_proto_to_binary(save_path)
            print('Merge successful, merged model has been written to ' + save_path)
    else:
        save_path = save_name + '.trontxt'
        tron_net.write_proto_to_txt(save_path)
        print('Convert successful, model has been written to ' + save_path)
        if args.merge_op:
            save_path = save_name + '_merged.trontxt'
            merged_net.write_proto_to_txt(save_path)
            print('Merge successful, merged model has been written to ' + save_path)
