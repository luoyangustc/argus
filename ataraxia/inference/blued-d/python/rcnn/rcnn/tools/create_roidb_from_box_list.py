'''
this file is split from imdb.py [function:create_roidb_from_box_list]. For memory limit, create roidb for imagenet can 
not be done in one shot. This file aims to solve this problem by split the imagenet dataset by several parts, and create 
roidb separately.
'''
from __future__ import print_function
import os
import cPickle
import numpy as np
import argparse
from ..processing.bbox_transform import bbox_overlaps
from ..dataset import *
import gc
from memory_profiler import profile
from multiprocessing import Process

#@profile(precision=4)
def create_roidb_from_box_list(box_list, gt_roidb):
    """
    given ground truth, prepare roidb
    :param box_list: [image_index] ndarray of [box_index][x1, x2, y1, y2]
    :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    """
    num_images = len(gt_roidb)
    num_classes = 201
    assert len(box_list) == num_images, 'number of boxes matrix must match number of images'
    roidb = []
    for i in xrange(num_images):
        print("create roidb from box list:", i)
        roi_rec = dict()
        roi_rec['image'] = gt_roidb[i]['image']
        roi_rec['height'] = gt_roidb[i]['height']
        roi_rec['width'] = gt_roidb[i]['width']
        print("image:{},height:{},width:{}".format(roi_rec['image'], roi_rec['height'], roi_rec['width']))
        boxes = box_list[i]
        #print(boxes)
        if boxes.shape[1] == 5:
            boxes = boxes[:, :4]
        num_boxes = boxes.shape[0]
        #print("num_boxes:{},num gt_boxes:{}".format(num_boxes, gt_roidb[i]['boxes'].size))
        overlaps = np.zeros((num_boxes, num_classes), dtype=np.float32)
        if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
            gt_boxes = gt_roidb[i]['boxes']
            gt_classes = gt_roidb[i]['gt_classes']
            # n boxes and k gt_boxes => n * k overlap
            gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
            # for each box in n boxes, select only maximum overlap (must be greater than zero)
            argmaxes = gt_overlaps.argmax(axis=1)
            maxes = gt_overlaps.max(axis=1)
            #del gt_overlaps
            I = np.where(maxes > 0)[0]
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

        roi_rec.update({'boxes': boxes,
                        'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                        #'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})

        # background roi => background class
        zero_indexes = np.where(roi_rec['max_overlaps'] == 0)[0]
        assert all(roi_rec['max_classes'][zero_indexes] == 0)
        # foreground roi => foreground class
        nonzero_indexes = np.where(roi_rec['max_overlaps'] > 0)[0]
        assert all(roi_rec['max_classes'][nonzero_indexes] != 0)

        roidb.append(roi_rec)
        #del overlaps
        #gc.collect()
    print("create complete")

    return roidb


def load_rpn_data(root_path, name, full=False):
    if full:
        rpn_file = os.path.join(root_path, 'rpn_data', name + '_full_rpn.pkl')
    else:
        rpn_file = os.path.join(root_path, 'rpn_data', name + '_rpn.pkl')
    print('loading {}'.format(rpn_file))
    assert os.path.exists(rpn_file), 'rpn data not found at {}'.format(rpn_file)
    with open(rpn_file, 'rb') as f:
        box_list = cPickle.load(f)
    return box_list


def rpn_roidb(gt_roidb, rpn_load_roidb, num_images_start, num_images_end, append_gt=False):
    """
    get rpn roidb and ground truth roidb
    :param gt_roidb: ground truth roidb
    :param num_images_start, num_images_end: images range from num_images_start to num_images_end
    :param append_gt: append ground truth
    :return: roidb of rpn
    """
    gt_roidb_split = gt_roidb[num_images_start:num_images_end]
    rpn_roidb_split = rpn_load_roidb[num_images_start:num_images_end]
    rpn_roidb_split = create_roidb_from_box_list(rpn_roidb_split, gt_roidb_split)
    if append_gt:
        print('appending ground truth annotations')
        rpn_roidb_split = merge_roidbs(gt_roidb_split, rpn_roidb_split)
    del gt_roidb_split
    gc.collect()
    return rpn_roidb_split


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        num_split, proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    gt_roidb = imdb.gt_roidb()
    rpn_load_roidb = load_rpn_data(root_path, dataset_name)
    num_images = len(gt_roidb)

    for split in xrange(num_split):
        p = Process(target=func, args=(num_images, num_split, split, proposal, gt_roidb, rpn_load_roidb, append_gt, flip))
        p.start()
        p.join()


def func(num_images, num_split, split, proposal, gt_roidb, rpn_load_roidb, append_gt, flip):
    num_images_start = num_images / num_split * split
    if split == num_split - 1:
        num_images_end = num_images
    else:
        num_images_end = num_images / num_split * (split + 1)

    roidb = eval(proposal + '_roidb')(gt_roidb, rpn_load_roidb, num_images_start, num_images_end, append_gt)

    if flip:
        roidb = imdb.append_flipped_images(roidb)

    cachefile = str(split) + '.pkl'
    print("save to cache:", cachefile)
    with open(cachefile, 'w') as f:
        cPickle.dump(roidb, f, protocol=cPickle.HIGHEST_PROTOCOL)

    #del roidb
    #gc.collect()


def merge_roidbs(a, b):
    """
    merge roidbs into one
    :param a: roidb to be merged into
    :param b: roidb to be merged
    :return: merged imdb
    """
    assert len(a) == len(b)
    for i in xrange(len(a)):
        a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
        a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'], b[i]['gt_classes']))
        # a[i]['gt_overlaps'] = np.vstack((a[i]['gt_overlaps'], b[i]['gt_overlaps']))
        a[i]['max_classes'] = np.hstack((a[i]['max_classes'], b[i]['max_classes']))
        a[i]['max_overlaps'] = np.hstack((a[i]['max_overlaps'], b[i]['max_overlaps']))
    return a


def split_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        proposal='rpn', append_gt=True, flip=False, num_split=4):

    load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        num_split=num_split, proposal='rpn', append_gt=True, flip=False)

    rois_all = []
    for split in xrange(num_split):
        cachefile = str(split) + '.pkl'
        with open(cachefile, 'rb') as f:
            box_list = cPickle.load(f)
        rois_all.extend(box_list)
        print("num_images in this split:{}, num_images all:{}".format(len(box_list), len(rois_all)))

    rois_all_file = 'imagenet.pkl'
    print("save all to:",rois_all_file )
    with open(rois_all_file, 'w') as f:
        cPickle.dump(rois_all, f, protocol=cPickle.HIGHEST_PROTOCOL)

def parse_args():
    parser = argparse.ArgumentParser(description='Post processing')

    parser.add_argument('--dataset_name', help='dataset name', default='imagenet', type=str)
    parser.add_argument('--image_set_name', help='image_set name', default='train', type=str)
    parser.add_argument('--root_path', help='output data folder', default='data', type=str)
    parser.add_argument('--dataset_path', help='dataset path', default='data/imagenet', type=str)
    parser.add_argument('--proposal', help='proposal method',default='rpn', type=str)
    parser.add_argument('--append_gt', help='append gt or not',action='store_true')
    parser.add_argument('--flip', help='flip or not',action='store_true')
    parser.add_argument('--num_split', help='num of split to merge',default=10, type=int)

    args = parser.parse_args()
    return args

#@profile(precision=4)
def main():
    args = parse_args()
    print(args)
    split_roidb(args.dataset_name, args.image_set_name, args.root_path, args.dataset_path,
                args.proposal, args.append_gt, args.flip, args.num_split)

if __name__ == '__main__':
    main()


