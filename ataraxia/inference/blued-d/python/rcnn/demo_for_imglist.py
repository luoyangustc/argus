from __future__ import print_function
import argparse
import os
import cv2
import cPickle
import mxnet as mx
import numpy as np
from rcnn.config import config, default, generate_config
from rcnn.symbol import *
from rcnn.io.image import resize, transform
from rcnn.core.tester import Predictor, im_detect, im_proposal, vis_all_detection, draw_all_detection
from rcnn.utils.load_model import load_param
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from rcnn.processing.box_voting import py_box_voting_wrapper

CLASSES = ('__background__',\
         'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo', 'artichoke',\
         'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam', 'banana', 'band_aid',\
         'banjo', 'baseball', 'basketball', 'bathing_cap', 'beaker', 'bear', 'bee',\
         'bell_pepper', 'bench', 'bicycle', 'binder', 'bird', 'bookshelf', 'bow_tie',\
         'bow', 'bowl', 'brassiere', 'burrito', 'bus', 'butterfly', 'camel', 'can_opener',\
         'car', 'cart', 'cattle', 'cello', 'centipede', 'chain_saw', 'chair', 'chime',\
         'cocktail_shaker', 'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',\
         'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper', 'digital_clock',\
         'dishwasher', 'dog', 'domestic_cat', 'dragonfly', 'drum', 'dumbbell', 'electric_fan',\
         'elephant', 'face_powder', 'fig', 'filing_cabinet', 'flower_pot', 'flute', 'fox',\
         'french_horn', 'frog', 'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',\
         'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger', 'hammer', 'hamster',\
         'harmonica', 'harp', 'hat_with_a_wide_brim', 'head_cabbage', 'helmet', 'hippopotamus',\
         'horizontal_bar', 'horse', 'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',\
         'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard', 'lobster', 'maillot',\
         'maraca', 'microphone', 'microwave', 'milk_can', 'miniskirt', 'monkey', 'motorcycle',\
         'mushroom', 'nail', 'neck_brace', 'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener',\
         'perfume', 'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza', 'plastic_bag',\
         'plate_rack', 'pomegranate', 'popsicle', 'porcupine', 'power_drill', 'pretzel', 'printer', 'puck',\
         'punching_bag', 'purse', 'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator', 'remote_control',\
         'rubber_eraser', 'rugby_ball', 'ruler', 'salt_or_pepper_shaker', 'saxophone', 'scorpion',\
         'screwdriver', 'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile', 'snowplow',\
         'soap_dispenser', 'soccer_ball', 'sofa', 'spatula', 'squirrel', 'starfish', 'stethoscope',\
         'stove', 'strainer', 'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',\
         'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie', 'tiger', 'toaster',\
         'traffic_light', 'train', 'trombone', 'trumpet', 'turtle', 'tv_or_monitor', 'unicycle', 'vacuum',\
         'violin', 'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft', 'whale', 'wine_bottle',\
         'zebra')

config.TEST.HAS_RPN = True
SHORT_SIDE = config.SCALES[0][0]
LONG_SIDE = config.SCALES[0][1]
PIXEL_MEANS = config.PIXEL_MEANS
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
LABEL_SHAPES = None
# visualization
CONF_THRESH = 1e-3
NMS_THRESH = 0.3
nms = py_nms_wrapper(NMS_THRESH)
box_voting = py_box_voting_wrapper(config.TEST.BOX_VOTING_IOU_THRESH, config.TEST.BOX_VOTING_SCORE_THRESH,
                                   with_nms=True)


def get_net(symbol, prefix, epoch, ctx):
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(DATA_SHAPES)
    arg_names, aux_names = symbol.list_arguments(), symbol.list_auxiliary_states()
    arg_shape, _, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    aux_shape_dict = dict(zip(aux_names, aux_shape))

    # check shapes
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    predictor = Predictor(symbol, DATA_NAMES, LABEL_NAMES, context=ctx,
                          provide_data=DATA_SHAPES, provide_label=LABEL_SHAPES,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor


def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [mx.nd.array(im_array), mx.nd.array(im_info)]
    data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch, DATA_NAMES, im_scale


def image_path_from_index(index, dataset_path, image_set):
    """
    given image index, find out full path
    :param index: index of a specific image
    :return: full path of this image
    """
    image_file = os.path.join(dataset_path, 'DET', 'Data','DET', image_set, index + '.JPEG')
    assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
    return image_file

def demo_net(predictor, dataset, image_set,
             root_path, dataset_path, vis=False, vis_image_dir=None, use_box_voting=False):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :param vis: will save as a new image if not visualized
    :return: None
    """
    image_set_index_file = os.path.join(dataset_path, 'DET', 'ImageSets', 'DET', image_set + '.txt')
    assert os.path.exists(image_set_index_file), image_set_index_file + ' not found'
    with open(image_set_index_file) as f:
        image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]

    num_images = len(image_set_index)
    num_classes = len(CLASSES)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    i=0
    for index in image_set_index:
        image_file = image_path_from_index(index, dataset_path, image_set)
        print("processing {}/{} image:{}".format(i, num_images, image_file))
        im = cv2.imread(image_file)
        data_batch, data_names, im_scale = generate_batch(im)
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, im_scale)
        for cls in CLASSES:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where(cls_scores >= CONF_THRESH)[0]
            cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(cls_dets)

            # apply box voting after nms
            if use_box_voting:
                nms_cls_dets = cls_dets[keep, :]
                all_boxes[cls_ind][i] = box_voting(nms_cls_dets, cls_dets)
            else:
                all_boxes[cls_ind][i] = cls_dets[keep, :]

        boxes_this_image = [[]] + [all_boxes[j][i] for j in xrange(1, len(CLASSES))]

        # print results
        """
        print('class ---- [[x1, x2, y1, y2, confidence]]')
        for ind, boxes in enumerate(boxes_this_image):
            if len(boxes) > 0:
                print('---------', CLASSES[ind], '---------')
                print(boxes)
        """

        i+=1
        if vis:
            #vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
            if not os.path.exists(vis_image_dir):
                os.mkdir(vis_image_dir)
            result_file = os.path.join(vis_image_dir, index.strip().split('/')[-1] + '_result' + '.JPEG')
            print('results saved to %s' % result_file)
            im = draw_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cv2.imwrite(result_file, im)
    
    print("num of images: detection:{}, gt:{}".format(len(all_boxes[0]), num_images))
    # assert len(all_boxes) == num_images, 'calculations not complete'

    # save results
    cache_folder = os.path.join(root_path, 'cache')
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    cache_file = os.path.join(cache_folder, dataset + '_' + image_set + '_detections.pkl')
    print("save to {}".format(cache_file))
    with open(cache_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network')
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--prefix', help='saved model prefix', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', type=int)
    parser.add_argument('--gpu', help='GPU device to use', default=0, type=int)
    parser.add_argument('--vis', help='display result', action='store_true')
    parser.add_argument('--vis_image_dir', help='if vis, image results are saved in this folder', default='data/vis', type=str)
    parser.add_argument('--use_global_context', help='use roi global context for classification', action='store_true')
    parser.add_argument('--use_roi_align', help='replace ROIPooling with ROIAlign', action='store_true')
    parser.add_argument('--use_box_voting', help='use box voting in test', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ctx = mx.gpu(args.gpu)

    if args.use_global_context or args.use_roi_align:
        symbol = eval('get_' + args.network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS,
                                               use_global_context=args.use_global_context, use_roi_align=args.use_roi_align)
    else:
        symbol = eval('get_' + args.network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)

    predictor = get_net(symbol, args.prefix, args.epoch, ctx)
    demo_net(predictor, args.dataset, args.image_set,
             args.root_path, args.dataset_path, args.vis, args.vis_image_dir, args.use_box_voting)


if __name__ == '__main__':
    main()
