#!/bin/usr/env python

# by zhaoyafei0210@gmail.com

import os
import os.path as osp

import numpy as np
from numpy.linalg import norm
# import scipy.io as sio
import cv2

import json
import time

from easydict import EasyDict as edict

import _init_paths
#from compare_feats import calc_similarity_cosine


try:
    import mxnet as mx
except ImportError as err:
    raise ImportError('{}. Please set the correct mxnet_root in {} '
                      'or in the first line of your main python script.'.format(
                          err, osp.abspath(osp.dirname(__file__)) + '/_init_paths.py')
                      )


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InitError(Error):
    """ Class for Init exceptions in this module."""
    pass


class LoadDataError(Error):
    """Exception for load image data"""
    pass


class FeatureLayerError(Error):
    """Exception for Invalid feature layer names."""
    pass


class ExtractionError(Error):
    """Exception from extract_xxx()."""
    pass


class MxnetFeatureExtractor(object):
    def __init__(self, config_json):
        self.net = None
#        self.net_blobs = None
        self.image_shape = None
        self.batch_size = None
        self.net_ctx = mx.cpu()
        self.mean_arr = None
        self.input_blob = None

        self.config = {
            #"network_symbols": "/path/to/prototxt",
            #"network_params": "/path/to/mxnetmodel",
            #"data_mean": "/path/to/the/mean/file",
            "data_mean": "",
            #"feature_layer": "fc5",
            "batch_size": 1,
            "input_width": 112,
            "input_height": 112,
            "input_scale": 1.0,
            # "raw_scale": 1.0,
            # default is BGR, be careful of your input image"s channel
            "channel_swap": (0, 1, 2),
            "image_as_grey": 0,
            # 0,None - will not use mirror_trick, 1 - eltavg (i.e.
            # eltsum()*0.5), 2 - eltmax, 3 - concat
            "mirror_trick": 0,
            "normalize_output": False,
            "cpu_only": 0,
            "gpu_id": 0
        }

        if isinstance(config_json, str):
            if osp.isfile(config_json):
                fp = open(config_json, 'r')
                _config = json.load(fp)
                fp.close()
            else:
                _config = json.loads(config_json)
        else:
            _config = config_json

        # must convert to str, because json.load() outputs unicode which is not support
        # in mxnet's cpp function
        _config['network_model'] = str(_config['network_model'])
        # _config['network_symbols'] = str(_config['network_symbols'])
        # _config['network_params'] = str(_config['network_params'])
        # _config['data_mean'] = str(_config['data_mean'])
        _config['feature_layer'] = str(_config['feature_layer'])
        _config['channel_swap'] = tuple(
            [int(i.strip()) for i in _config['channel_swap'].split(',')])

        self.config.update(_config)
        # print('===> network configs:\n', self.config)

        data_mean = self.config.get('data_mean', None)
        if (data_mean):
            if isinstance(data_mean, list):
                # mean_arr = np.matrix(self.config['data_mean']).A1
                mean_arr = np.array(self.config['data_mean'], dtype=np.float32)
            elif data_mean.endswith('.npy'):
                mean_arr = np.load(str(data_mean))
                if mean_arr is None:
                    raise InitError('Failed to load "data_mean": ' +
                                    str(data_mean))
            else:
                raise InitError(
                    'data_mean must be a valid path to .npy or a list of 3 floats')

            self.mean_arr = mean_arr
            # print 'mean array shape: ', self.mean_arr.shape
            # print 'mean array: ', self.mean_arr

        if (int(self.config['mirror_trick']) not in [0, 1, 2, 3]):
            raise InitError('"mirror_trick" must be one from [0,1,2,3]')

        # print'\n===> MxnetFeatureExtractor.config: \n', self.config

        if(not self.config['cpu_only'] and self.config['gpu_id'] >= 0):
            # print '===> Using GPU context'
            self.net_ctx = mx.gpu(self.config['gpu_id'])

        # if (self.config['feature_layer'] not in self.net.layer_dict.keys()):
        #     raise FeatureLayerError('Invalid feature layer names: '
        #                             + self.config['feature_layer'])
#        self.config['feature_layer'] = self.get_feature_layers(
#            self.config['feature_layer'])

#        self.net_blobs = OrderedDict([(k, v.data)
#                                  for k, v in self.net.blobs.items()])
#        # print'self.net_blobs: ', self.net_blobs
#        for k, v in self.net.blobs.items():
#            # printk, v

        self.image_shape = (
            self.config['input_height'], self.config['input_width'], 3)
        self.batch_size = self.config['batch_size']
        # print'---> batch size in the config: ', self.batch_size

        if self.config['mirror_trick'] > 0:
            # print'---> need to double the batch size of the net input data
            # because of mirror_trick'
            final_batch_size = self.batch_size * 2
        else:
            final_batch_size = self.batch_size

        data_shape = (final_batch_size, 3,
                      self.config['input_height'], self.config['input_width'])

        vec = self.config['network_model'].split(',')
        if len(vec) < 2:
            raise InitError(
                'network_model must be in the form of "prefix,epoch"')

        prefix = vec[0]
        epoch = int(vec[1])
        # print 'model prefix: ', prefix
        # print 'model epoch: ', epoch

        net = edict()
        net.ctx = self.net_ctx
        net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(
            prefix, epoch)

        all_layers = net.sym.get_internals()
        net.sym = all_layers[self.config['feature_layer']]
        # print net.sym.get_internals()
        net.model = mx.mod.Module(
            symbol=net.sym, context=net.ctx, label_names=None)
        net.model.bind(
            data_shapes=[('data', data_shape)])
        net.model.set_params(net.arg_params, net.aux_params)

        self.net = net
        self.input_blob = np.zeros(data_shape, dtype=np.float32)

    def get_batch_size(self):
        return self.batch_size

    def read_image(self, img_path):
        if self.config["image_as_grey"]:
            img = cv2.imread(img_path, 0)
        else:
            img = cv2.imread(img_path, 1)

        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print '---> img: ', img

        return img

    def preprocess(self, data):
        """
        Format input for network:
        - convert to single
        - resize to input dimensions (preserving number of channels)
        - transpose dimensions to K x H x W
        - reorder channels (for instance color to BGR)
        - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
        - subtract mean
        - scale feature

        Parameters
        ----------
        data : (H' x W' x K) ndarray

        Returns
        -------
        net_in : (K x H x W) ndarray for input to a Net
        """
        net_in = data.astype(np.float32, copy=False)
        # print 'net_in.shape: ', net_in.shape
        # print 'net_in: ', net_in

        channel_swap = self.config.get('channel_swap', None)
        # raw_scale = self.config.get('raw_scale', None)
        mean = self.mean_arr
        input_scale = self.config.get('input_scale', None)

        if channel_swap is not None:
            net_in = net_in[:, :, channel_swap]
            # print 'net_in after channel_swap: ', net_in
        # if raw_scale is not None:
        #     net_in *= raw_scale
        if mean is not None:
            net_in -= mean
            # print 'net_in after removing mean: ', net_in
        if input_scale is not None:
            net_in *= input_scale
            # print 'net_in after input_scale: ', net_in

        net_in = net_in.transpose((2, 0, 1))
        # print 'net_in.shape after transpose: ', net_in.shape
        # print 'net_in after transpose: ', net_in
        return net_in

    def load_image_to_data_buffer(self, img, load_idx=0):
        if img.shape != self.image_shape:
            raise LoadDataError('image shape must be : ', self.image_shape)

        if load_idx + 1 > self.batch_size:
            raise LoadDataError(
                'Must have load_idx < batch size = ', self.batch_size)

        # if self.config['channel_swap'] != (0, 1, 2):
        #     #print('to rgb')
        #     img = img[..., self.config['channel_swap']]

        # if self.mean_arr is not None:
        #     # v_mean=np.array([127.5, 127.5, 127.5],
        #     #                 dtype=np.float32).reshape((1, 1, 3))
        #     img = img.astype(np.float32) - self.mean_arr
        #     # img *= 0.0078125

        # if self.config['input_scale'] != 1.0:
        #     img *= self.config['input_scale']

        # self.input_blob[load_idx] = np.transpose(img, (2, 0, 1))
        self.input_blob[load_idx] = self.preprocess(img)
        # print '---> load_idx: ', load_idx
        # print '---> self.input_blob[load_idx]: ', self.input_blob[load_idx]

        if self.config['mirror_trick'] > 0:
            n_channels = self.input_blob.shape[1]
            for j in range(n_channels):
                self.input_blob[load_idx + self.batch_size][j][...] = np.fliplr(
                    self.input_blob[load_idx][j][...])
        # print '---> self.input_blob[load_idx+batch_size]: ',
        # self.input_blob[load_idx+ self.batch_size]

    def get_features(self, n_imgs=None):
        if not n_imgs:
            n_imgs = self.batch_size

        data = mx.nd.array(self.input_blob)
        # print '---> data[0]: ', data[0]
        # print '---> data[batch_size]: ', data[self.batch_size]

        db = mx.io.DataBatch(data=(data,))
        # print '---> db.data[0]: ', db.data[0][0]
        # print '---> db.data[batch_size]: ', db.data[0][self.batch_size]

        features = []

        self.net.model.forward(db, is_train=False)
        outputs = self.net.model.get_outputs()[0]
        # print('outputs.shape: ', outputs.shape)
        # print('outputs: ', outputs)

        for j in range(n_imgs):
            embedding = outputs[j].asnumpy().flatten()
            # print '---> embedding.shape: ', embedding.shape
            # print '---> embedding: ', embedding

            if self.config['mirror_trick'] > 0:
                embedding_flip = outputs[j +
                                         self.batch_size].asnumpy().flatten()
                # print '---> embedding_flip.shape: ', embedding_flip.shape
                # print '---> embedding_flip: ', embedding_flip

                # sim = calc_similarity_cosine(embedding, embedding_flip)
                # print('---> flip_sim=%f\n' % sim)

                if self.config['mirror_trick'] == 1:
                    # print '---> elt_avg embedding and embedding_flip'
                    embedding += embedding_flip
                    embedding *= 0.5
                elif self.config['mirror_trick'] == 2:
                    # print '---> elt_max embedding and embedding_flip'
                    embedding = np.maximum(embedding, embedding_flip)
                else:
                    # print '---> concat embedding and embedding_flip'
                    embedding = np.concatenate([embedding, embedding_flip])
            # print '---> after mirror_trick, embedding.shape: ', embedding.shape
            # print '---> after mirror_trick, embedding: ', embedding

            if self.config['normalize_output'] > 0:
                _norm = np.linalg.norm(embedding)
                if _norm > 0:
                    embedding /= _norm

            # print '---> after norm, embedding.shape: ', embedding.shape
            # print '---> after norm, embedding: ', embedding

            features.append(embedding)

        return features

    def extract_feature(self, image):
        if isinstance(image, str):
            image = self.read_image(image)

        self.load_image_to_data_buffer(image)
        feature = self.get_features(1)[0]

        return feature

    def extract_features_batch(self, images):
        n_imgs = len(images)

        if (n_imgs > self.batch_size):
            raise ExtractionError(
                'Number of input images > batch_size=', self.batch_size)

        load_idx = 0

        for img in images:
            self.load_image_to_data_buffer(img, load_idx)

            load_idx += 1

        # cnt_predict = 0
        # time_predict = 0.0

        # t1 = time.clock()

        features = self.get_features(n_imgs)

        # t2 = time.clock()
        # time_predict += (t2 - t1)
        # cnt_predict += n_imgs

        return features

    def extract_features_for_image_list(self, image_list, img_root_dir=None):
        # cnt_load_img = 0
        # time_load_img = 0.0
        # cnt_predict = 0
        # time_predict = 0.0
        img_batch = []
        features = []

        for cnt, path in enumerate(image_list):
            # t1 = time.clock()

            if img_root_dir:
                path = osp.join(img_root_dir, path)

            img = self.read_image(path)
            # if cnt == 0:
            # print'image shape: ', img.shape

            img_batch.append(img)
            # t2 = time.clock()

            # cnt_load_img += 1
            # time_load_img += (t2 - t1)

            # # print'image shape: ', img.shape
            # # printpath, type(img), img.mean()
            if (len(img_batch) == self.batch_size or cnt + 1 == len(image_list)):
                _ftrs = self.extract_features_batch(img_batch)
                features.extend(_ftrs)
                img_batch = []

        # print('Load %d images, cost %f seconds, average time: %f seconds' %
        #       (cnt_load_img, time_load_img, time_load_img / cnt_load_img))
        # print '---> len(features): ', len(features)
        return features


if __name__ == '__main__':
    def load_image_list(list_file_name):
        # list_file_path = os.path.join(img_dir, list_file_name)
        f = open(list_file_name, 'r')
        img_fn_list = []

        for line in f:
            if line.startswith('#'):
                continue

            items = line.split()
            img_fn_list.append(items[0].strip())

        f.close()

        return img_fn_list

    config_json = './extractor_config.json'
    save_dir = './rlt_features'

    image_dir = r'../test_data/face_chips_112x112/'
    image_list_file = r'../test_data/face_chips_list.txt'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # test extract_features_for_image_list()
    save_name = 'img_list_features.npy'

    img_list = load_image_list(image_list_file)

    # init a feat_extractor, use a context to release mxnet objects
    print '\n===> init a feat_extractor'
    feat_extractor = MxnetFeatureExtractor(config_json)
    print('===> feat_extractor configs:\n', feat_extractor.config)

    print'\n===> test extract_features_for_image_list()'
    ftrs = feat_extractor.extract_features_for_image_list(img_list, image_dir)
#    np.save(osp.join(save_dir, save_name), ftrs)
    print '---> len(ftrs): ', len(ftrs)

    root_len = len(image_dir)

    for i in range(len(img_list)):
        spl = osp.split(img_list[i])
        base_name = spl[1]
#        sub_dir = osp.split(spl[0])[1]
        sub_dir = spl[0]

        if sub_dir:
            save_sub_dir = osp.join(save_dir,  sub_dir)
            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
        else:
            save_sub_dir = save_dir

        save_name = osp.splitext(base_name)[0] + '.npy'
        np.save(osp.join(save_sub_dir, save_name), ftrs[i])

    # test extract_feature()
    print '\n===> test extract_feature()'
    save_name_2 = 'single_feature.npy'

    ftr = feat_extractor.extract_feature(osp.join(image_dir, img_list[0]))
    np.save(osp.join(save_dir, save_name_2), ftr)

    ft_diff = ftr - ftrs[0]
    print '---> ftr', ftr
    print '---> sum(ft_diff): ', ft_diff.sum()
