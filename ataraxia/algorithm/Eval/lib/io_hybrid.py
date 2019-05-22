#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import logging
import mxnet as mx
import cv2
import numpy as np
from config import cfg


class empty_image(Exception):
    '''
    catch empty image error
    '''
    pass


def check_dir(path):
    '''
    '''
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        logging.info("{} does not exist, and it is now created.".format(dirname))
        os.mkdir(dirname)


def inst_iterators(data_train, data_dev, batch_size=1, data_shape=(3,224,224), resize=(-1,-1), resize_scale=(1,1), resize_area=(1,1), use_svm_label=False, use_dali=False):
    '''
    Instantiate specified training and developing data iterators
    :params:
    data_train      training rec/lst
    data_dev        developing rec/lst
    batch_size      mini batch size, sum of all device
    data_shape      input shape
    resize          resize shorter edge of (train,dev) data, -1 means no resize
    resize_scale    resize train-data into (width*s, height*s), with s randomly chosen from this range 
    resize_area     Change the area (namely width * height) to a random value in [min_random_area, max_random_area]. Ignored if random_resized_crop is False 
    use_svm_label   set as True if classifier needs svm label name
    use_dali        set as True if nvidia dali is supposed to be used
    :return:
    train, dev      tuple of 2 iterators
    '''
    # initialization
    assert data_train and data_dev, logging.error("Please input training or developing data") 
    mean, std = cfg.TRAIN.MEAN_RGB, cfg.TRAIN.STD_RGB 
    assert len(mean)==3 and len(std)==3, logging.error("Mean or Std should be a list of 3 items")
    mean_r, mean_g, mean_b, std_r, std_g, std_b = mean[:] + std[:] 
    min_random_scale, max_random_scale = resize_scale 
    min_random_area, max_random_area = resize_area 
    min_aspect_ratio = cfg.TRAIN.MIN_ASPECT_RATIO if cfg.TRAIN.MIN_ASPECT_RATIO else None
    logging.info('Input normalization : Mean-RGB {}, Std-RGB {}'.format([mean_r, mean_g, mean_b],[std_r, std_g, std_b]))
    logging.info('Input scale augmentation : Max-random-sclae {}, Min-random-scale {}'.format(max_random_scale, min_random_scale))
    logging.info('Input area augmentation : Max-random-area {}, Min-random-area {}'.format(max_random_area, min_random_area))
    resize_train, resize_dev = resize
    label_name = 'softmax_label' if not use_svm_label else 'svm_label'

    # build iterators
    if not cfg.TRAIN.USE_DALI and cfg.TRAIN.USE_REC:
        logging.info("Creating recordio iterators")
        train = mx.io.ImageRecordIter(
                dtype               = cfg.TRAIN.DATA_TYPE,
                path_imgrec         = data_train,
                preprocess_threads  = cfg.TRAIN.PROCESS_THREAD,
                data_name           = 'data',
                label_name          = label_name,
                label_width         = cfg.TRAIN.LABEL_WIDTH,
                data_shape          = data_shape,
                batch_size          = batch_size,
                resize              = resize_train,
                max_random_scale    = max_random_scale,
                min_random_scale    = min_random_scale,
                shuffle             = cfg.TRAIN.SHUFFLE,
                rand_crop           = cfg.TRAIN.RAND_CROP,
                rand_mirror         = cfg.TRAIN.RAND_MIRROR,
                max_rotate_angle    = cfg.TRAIN.MAX_ROTATE_ANGLE,
                max_aspect_ratio    = cfg.TRAIN.MAX_ASPECT_RATIO,
                min_aspect_ratio    = min_aspect_ratio, 
                random_resized_crop = cfg.TRAIN.RANDOM_RESIZED_CROP,
                max_random_area     = max_random_area,
                min_random_area     = min_random_area,
                max_img_size        = cfg.TRAIN.MAX_IMG_SIZE,
                min_img_size        = cfg.TRAIN.MIN_IMG_SIZE,
                max_shear_ratio     = cfg.TRAIN.MAX_SHEAR_RATIO,
                brightness          = cfg.TRAIN.BRIGHTNESS_JITTER,
                contrast            = cfg.TRAIN.CONTRAST_JITTER,
                saturation          = cfg.TRAIN.SATURATION_JITTER,
                hue                 = cfg.TRAIN.HUE_JITTER,
                pca_noise           = cfg.TRAIN.PCA_NOISE,
                random_h            = cfg.TRAIN.RANDOM_H,
                random_s            = cfg.TRAIN.RANDOM_S,
                random_l            = cfg.TRAIN.RANDOM_L,
                mean_r              = mean_r,
                mean_g              = mean_g,
                mean_b              = mean_b,
                std_r               = std_r,
                std_g               = std_g,
                std_b               = std_b,
                inter_method        = cfg.TRAIN.INTERPOLATION_METHOD
                )
        dev = mx.io.ImageRecordIter(
                dtype               = cfg.TRAIN.DATA_TYPE,
                path_imgrec         = data_dev,
                preprocess_threads  = cfg.TRAIN.PROCESS_THREAD,
                data_name           = 'data',
                label_name          = label_name,
                label_width         = cfg.TRAIN.LABEL_WIDTH,
                batch_size          = batch_size,
                data_shape          = data_shape,
                resize              = resize_dev,
                shuffle             = False,
                rand_crop           = False,    # center crop
                rand_mirror         = False,
                mean_r              = mean_r,
                mean_g              = mean_g,
                mean_b              = mean_b,
                std_r               = std_r,
                std_g               = std_g,
                std_b               = std_b,
                inter_method        = cfg.TRAIN.INTERPOLATION_METHOD
                )

    elif not cfg.TRAIN.USE_DALI and not cfg.TRAIN.USE_REC:
        logging.info("Creating image iterators")
        # set decoding thread number
        os.environ['MXNET_CPU_WORKER_NTHREADS'] = str(cfg.TRAIN.PROCESS_THREAD) 
        # set rand_crop and rand_resize as default, and append separately
        aug_list_train = mx.image.CreateAugmenter(
                data_shape          = data_shape,
                resize              = resize_train,
                rand_mirror         = cfg.TRAIN.RAND_MIRROR,
                mean                = np.asarray(mean),
                std                 = np.asarray(std),
                brightness          = cfg.TRAIN.BRIGHTNESS_JITTER,
                contrast            = cfg.TRAIN.CONTRAST_JITTER,
                saturation          = cfg.TRAIN.SATURATION_JITTER,
                hue                 = cfg.TRAIN.HUE_JITTER,
                pca_noise           = cfg.TRAIN.PCA_NOISE,
                inter_method        = cfg.TRAIN.INTERPOLATION_METHOD
                )
        
        if cfg.TRAIN.RAND_CROP and min_random_scale != 1: 
            aug_list_train.append(mx.image.RandomSizedCropAug(
                    (data_shape[2],data_shape[1]), 
                    min_random_scale**2, 
                    (1-cfg.TRAIN.MAX_ASPECT_RATIO, 1+cfg.TRAIN.MAX_ASPECT_RATIO), 
                    cfg.TRAIN.INTERPOLATION_METHOD)) 
        elif cfg.TRAIN.RAND_CROP:
            aug_list_train.append(mx.image.RandomCropAug(
                    (data_shape[2],data_shape[1]), 
                    cfg.TRAIN.INTERPOLATION_METHOD))

        # set rand_crop and rand_resize as default to use center-crop
        aug_list_dev = mx.image.CreateAugmenter(
                data_shape          = data_shape,
                resize              = resize_dev,
                mean                = np.asarray(mean),
                std                 = np.asarray(std),
                inter_method        = cfg.TRAIN.INTERPOLATION_METHOD
                )
                
        train = mx.image.ImageIter(
                dtype               = cfg.TRAIN.DATA_TYPE,
                path_imglist        = data_train,
                data_name           = 'data',
                label_name          = label_name,
                label_width         = cfg.TRAIN.LABEL_WIDTH,
                data_shape          = data_shape,
                batch_size          = batch_size,
                path_root           = cfg.TRAIN.TRAIN_IMG_PREFIX,
                shuffle             = cfg.TRAIN.SHUFFLE,
                last_batch_handle   = cfg.TRAIN.LAST_BATCH_HANDLE,
                aug_list            = aug_list_train
                )
        dev = mx.image.ImageIter(
                dtype               = cfg.TRAIN.DATA_TYPE,
                path_imglist        = data_dev,
                data_name           = 'data',
                label_name          = label_name,
                label_width         = cfg.TRAIN.LABEL_WIDTH,
                data_shape          = data_shape,
                batch_size          = batch_size,
                path_root           = cfg.TRAIN.DEV_IMG_PREFIX,
                shuffle             = cfg.TRAIN.SHUFFLE,
                last_batch_handle   = cfg.TRAIN.LAST_BATCH_HANDLE,
                aug_list            = aug_list_dev
                )

    elif cfg.TRAIN.USE_DALI and cfg.TRAIN.USE_REC:
        from dali_util import HybridTrainPipe, HybridValPipe
        from nvidia.dali.plugin.mxnet import DALIClassificationIterator 
        num_gpus = len(cfg.TRAIN.GPU_IDX)
        batch_size /= num_gpus
        train_pipes = [HybridTrainPipe(batch_size=batch_size, num_threads=cfg.TRAIN.PROCESS_THREAD, device_id = i, num_gpus = num_gpus) for i in range(num_gpus)]
        dev_pipes = [HybridValPipe(batch_size=batch_size, num_threads=cfg.TRAIN.PROCESS_THREAD, device_id = i, num_gpus = num_gpus) for i in range(num_gpus)]
        train_pipes[0].build()
        dev_pipes[0].build()
        train = DALIClassificationIterator(train_pipes, train_pipes[0].epoch_size("Reader"))
        dev = DALIClassificationIterator(dev_pipes, dev_pipes[0].epoch_size("Reader"))

    else:
        logging.error('Invalid data loader type')
        pass
    logging.info("Data iters created successfully")
    return train, dev 


def np_img_preprocessing(img, as_float=True, **kwargs):
    '''
    '''
    assert isinstance(img, np.ndarray), logging.error("Input images should be type of numpy.ndarray")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        img = img.astype(float)
    # reshape
    if 'resize_w_h' in kwargs and not kwargs['keep_aspect_ratio']:
        img = cv2.resize(img, (kwargs['resize_w_h'][0], kwargs['resize_w_h'][1]))
    if 'resize_min_max' in kwargs and kwargs['keep_aspect_ratio']: 
        ratio = float(max(img.shape[:2]))/min(img.shape[:2])
        min_len, max_len = kwargs['resize_min_max']
        if min_len*ratio <= max_len or max_len == 0:    # resize by min
            if img.shape[0] > img.shape[1]:     # h > w
                img = cv2.resize(img, (min_len, int(min_len*ratio)))
            elif img.shape[0] <= img.shape[1]:   # h <= w
                img = cv2.resize(img, (int(min_len*ratio), min_len)) 
        elif min_len*ratio > max_len:   # resize by max
            if img.shape[0] > img.shape[1]:     # h > w
                img = cv2.resize(img, (int(max_len/ratio), max_len))
            elif img.shape[0] <= img.shape[1]:   # h <= w
                img = cv2.resize(img, (max_len, int(max_len/ratio))) 
    # normalization
    if 'mean_rgb' in kwargs:
        img -= kwargs['mean_rgb'][:]
    if 'std_rgb' in kwargs:
        img /= kwargs['std_rgb'][:] 
    # (h,w,c) => (c,h,w)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    logging.debug('img value: \n{}'.format(img))
    return img


def np_img_center_crop(img, crop_width):
    '''
    '''
    _, height, width = img.shape    # (c,h,w)
    assert (height >= crop_width and width >= crop_width), logging.error('crop size should be larger than image size')
    top = int(float(height) / 2 - float(crop_width) / 2)
    left = int(float(width) / 2 - float(crop_width) / 2)
    crop = img[:, top:(top + crop_width), left:(left + crop_width)]
    return crop


def np_img_multi_crop(img, crop_width, crop_number=3):
    _, height, width = img.shape
    assert crop_number in [3,5], logging.error('crop number should be 3 or 5 for now')
    assert (height >= crop_width and width >= crop_width), logging.error('image size should be larger than crop size')
    assert crop_width == min(height, width), logging.error('crop size should be equal to short edge for now')
    if height > width:
        cent_crop = np_img_center_crop(img, crop_width)
        top_crop = img[:, :crop_width, :]
        bottom_crop = img[:, -crop_width:, :]
        if crop_number == 3:
            return [top_crop, cent_crop, bottom_crop]
        elif crop_number == 5:
            stride = int((float(height)-float(crop_width))/4)
            top_crop_2 = img[:, stride:(stride + crop_width), :]
            bottom_crop_2 = img[:, (-crop_width - stride):-stride, :]
            return [top_crop, top_crop_2, cent_crop, bottom_crop_2, bottom_crop]
    elif width > height:
        cent_crop = np_img_center_crop(img, crop_width)
        left_crop = img[:, :, :crop_width]
        right_crop = img[:, :, -crop_width:]
        if crop_number == 3:
            return [left_crop, cent_crop, right_crop]
        elif crop_number == 5:
            stride = int((float(width)-float(crop_width))/4)
            left_crop_2 = img[:, :, stride:(stride + crop_width)]
            right_crop_2 = img[:, :, (-crop_width - stride):-stride]
            return [left_crop, left_crop_2, cent_crop, right_crop_2, right_crop]
    else:
        logging.info('input image gets 1:1 aspect ratio, return {} crops with exactly the same pixels'.format(crop_number))
        return[img for x in range(crop_number)]


def load_model(model_prefix, load_epoch, gluon_style=False):
    '''
    Load existing model
    :params:
    model_prefix        prefix of model with path
    load_epoch          which epoch to load
    gluon_style         set True to load model saved by gluon
    :return:
    sym, arg, aux       symbol, arg_params, aux_params of this model
                        aux_params will be an empty dict in gluon style
    '''
    assert model_prefix and load_epoch is not None, logging.error('Missing valid pretrained model prefix')
    assert load_epoch is not None, logging.error('Missing epoch of pretrained model to load')
    if not gluon_style:
        sym, arg, aux = mx.model.load_checkpoint(model_prefix, load_epoch)
    else:
        sym = mx.sym.load(model_prefix+'-symbol.json')
        save_dict = mx.nd.load('%s-%04d.params'%(model_prefix, load_epoch))
        arg, aux = dict(), dict()
        for k, v in save_dict.items():
            arg[k] = v
    logging.info('Loaded model: {}-{:0>4}.params'.format(model_prefix, load_epoch))

    return sym, arg, aux


def load_model_gluon(symbol, arg_params, aux_params, ctx, layer_name=None):
    '''
    Use to load net and params with gluon after load_model()
    '''
    def _init_gluon_style_params(raw_params, net_params, ctx):
        '''
        '''
        for param in raw_params:
            if param in net_params:
                net_params[param]._load_init(raw_params[param], ctx=ctx)
        return net_params

    if layer_name:
        net = symbol.get_internals()[layer_name + '_output']
    else:
        net = symbol
    net_hybrid = mx.gluon.nn.SymbolBlock(outputs=net, inputs=mx.sym.var('data'))
    net_params = net_hybrid.collect_params()
    net_params = _init_gluon_style_params(arg_params,net_params,ctx)
    net_params = _init_gluon_style_params(aux_params,net_params,ctx)

    return net_hybrid


def save_model(model_prefix, rank=0):
    '''
    '''
    assert model_prefix, logging.error('Model-prefix is needed to save model')
    dst_dir = os.path.dirname(model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d"%(model_prefix, rank))


def save_model_gluon(net, model_prefix, rank=0):
    '''
    '''
    net.collect_params().save("{}-{:0>4}.params".format(model_prefix, rank))
    net(mx.sym.Variable('data')).save("{}-symbol.json".format(model_prefix))
    logging.info('Saved checkpoint to \"{}-{:0>4}.params\"'.format(model_prefix, rank))
    return 0

def load_image_list(image_list_file):
    '''
    read image path file with syntax as follows(label is trivial):
    /path/to/image1.jpg (label)
    /path/to/image2.jpg (label)
    ...
    :params:

    :return:

    '''
    image_list = list()
    label_list = list()
    with open(image_list_file, 'r') as f:
        for buff in f:
            if len(buff.strip().split()) == 1:   # image path only
                image_list.append(buff.strip())
            elif len(buff.strip().split()) == 2:    # with labels
                image_list.append(buff.strip().split()[0])
                label_list.append(buff.strip().split()[1])
            else:
                logging.error("Image list syntax error!")
    return image_list, label_list


def load_category_list(cat_file, name_position=1, split=None):
    '''
    load category file
    file syntax:
    0 category_name
    1 category_name 
    :params:
    :return:
    '''
    tup_list = list()
    with open(cat_file,'r') as f:
        for buff in f.readlines():
            if ' ' in buff.strip():
                split = ' '
            elif ',' in buff.strip():
                split = ','
            tup_list.append(buff.strip().split(split)) 
    category_list = [tup[name_position] for tup in sorted(tup_list, key=lambda x:int(x[1-name_position]))]
    logging.debug(category_list)
    return category_list 
    
