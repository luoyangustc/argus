"""
usage:
python -m rcnn.tools.post_processing --pklfileIn data/cache/imagenet__val_detections1.pkl,data/cache/imagenet__val_detections2.pkl --pklfileOut a.pkl --dataset imagenet --image_set val --use_nms --use_box_voting
"""

from __future__ import print_function
import argparse
import os
import numpy as np
import mxnet as mx
from rcnn.config import config, default, generate_config_dataset
import cPickle
from ..symbol import *
from ..dataset import *
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from rcnn.processing.box_voting import py_box_voting_wrapper


def post_processing(pklfileIn, pklfileOut, dataset, image_set, root_path, dataset_path,
                    _use_nms=False, _use_box_voting=False):
    imdb = eval(dataset)(image_set, root_path, dataset_path)
    if _use_nms:
        nms = py_nms_wrapper(config.TEST.NMS)
    if _use_box_voting:
        box_voting = py_box_voting_wrapper(config.TEST.BOX_VOTING_IOU_THRESH,
                                           config.TEST.BOX_VOTING_SCORE_THRESH,
                                           with_nms=_use_nms)
    pklfileIn = pklfileIn.split(',')
    recs = []
    for detfile in pklfileIn:
        with open(detfile.strip(), 'r') as f:
            rec = cPickle.load(f)
            recs.append(rec)
    num_classes = imdb.num_classes
    num_images = imdb.num_images
    all_boxes = recs[0]

    # stack det result from different models
    for rec_idx in xrange(1, len(recs)):
        rec_per_model = recs[rec_idx]
        for cls_idx in xrange(num_classes):
            for img_idx in xrange(num_images):
                print("processing {}-th file:{} {}".format(rec_idx,cls_idx, img_idx))
                #print(all_boxes[cls_idx][img_idx], rec_per_model[cls_idx][img_idx])
                if len(rec_per_model[cls_idx][img_idx]) == 0:
                    continue
                elif len(all_boxes[cls_idx][img_idx]) == 0:
                    all_boxes[cls_idx][img_idx] = rec_per_model[cls_idx][img_idx]
                else:
                    all_boxes[cls_idx][img_idx]\
                        = np.concatenate((all_boxes[cls_idx][img_idx], rec_per_model[cls_idx][img_idx]), axis=0)

    # do nms/box voting
    for i in xrange(num_images):
        for j in xrange(1, num_classes):
            cls_dets = all_boxes[j][i]
            if cls_dets.size == 0:
                continue
            if _use_nms:
                keep = nms(cls_dets)
                if _use_box_voting:
                    nms_cls_dets = cls_dets[keep, :]
                    #print(nms_cls_dets)
                    all_boxes[j][i] = box_voting(nms_cls_dets, cls_dets)
                else:
                    all_boxes[j][i] = cls_dets[keep, :]
            else:
                if _use_box_voting:
                    all_boxes[j][i] = box_voting(cls_dets)
                # else: do nothing

    det_file = os.path.join(imdb.cache_path, pklfileOut)
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    imdb.evaluate_detections(all_boxes)
    return all_boxes


def parse_args():
    parser = argparse.ArgumentParser(description='Post processing')

    parser.add_argument('--pklfileIn', help='bbox results from different models, split by ,', type=str)
    parser.add_argument('--pklfileOut', help='merge bbox result', type=str)
    parser.add_argument('--dataset', help='dataset', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config_dataset(args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)

    parser.add_argument('--use_nms', help='use nms in fusing models', action='store_true')
    parser.add_argument('--use_box_voting', help='use box voting in fusing models', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    print(args)
    post_processing(args.pklfileIn, args.pklfileOut, args.dataset, args.image_set,
                    args.root_path, args.dataset_path,
                    args.use_nms, args.use_box_voting)

if __name__ == '__main__':
    main()


