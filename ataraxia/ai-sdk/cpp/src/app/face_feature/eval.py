#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import json
from multiprocessing import Process, Manager
import numpy as np
import requests
import struct
import os
import sys

from sklearn import metrics
from sklearn.model_selection import KFold
from scipy.optimize import brentq
from scipy import interpolate


def sorted_nparray_last_argmin(arr, min_val=None):
    if min_val is None:
        min_val = arr.min()

    for idx in range(len(arr)):
        if arr[idx] > min_val:
            min_idx = idx - 1
            break

    return min_idx


def sorted_nparray_first_argmax(arr, max_val=None):
    if max_val is None:
        max_val = arr.max()

    for idx in np.arange(len(arr) - 1, -1, -1):
        if arr[idx] < max_val:
            max_idx = idx + 1
            break

    return max_idx


def calc_accuracy(threshold, dist, actual_issame):
    """
        Calculate accuracy at some threshold
    """

    predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    tn = np.sum(np.logical_and(np.logical_not(
        predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc


def calc_val_far(threshold, dist, actual_issame):
    """
        Calculate VAL(=TPR) at some threshold
    """

    predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))

    false_accept = np.sum(np.logical_and(
        predict_issame, np.logical_not(actual_issame)))

    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)

    return val, far


def calc_roc(thresholds,
             dist,
             actual_issame):
    """
        ROC evaluation without k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    nrof_thresholds = len(thresholds)

    tprs = np.zeros(nrof_thresholds)
    fprs = np.zeros(nrof_thresholds)

    accuracies = np.zeros(nrof_thresholds)

    # Find the best threshold
    for idx, threshold in enumerate(thresholds):
        tprs[idx], fprs[idx], accuracies[idx] = calc_accuracy(
            threshold, dist, actual_issame)

    best_thresh_idx = np.argmax(accuracies)
    best_thresh = thresholds[best_thresh_idx]
    accuracy = accuracies[best_thresh_idx]

    val, far = calc_val_far(best_thresh, dist, actual_issame)

    best_accuracy = {
        "accuracy": accuracy,
        "threshold": best_thresh,
        "VAL": val,
        "FAR": far
    }

    return tprs, fprs, best_accuracy


def calc_val(thresholds,
             dist,
             actual_issame,
             far_targets=None):
    """
        Calculate VAL(=TPR) at target FARs without k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    if not far_targets:
        far_targets = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0]

    nrof_thresholds = len(thresholds)

    val_array = np.zeros(nrof_thresholds)
    far_array = np.zeros(nrof_thresholds)

    # Find the threshold that gives FAR = far_target
    for idx, threshold in enumerate(thresholds):
        val_array[idx], far_array[idx] = calc_val_far(
            threshold, dist, actual_issame)

    min_far_idx = sorted_nparray_last_argmin(far_array)

    max_far_idx = sorted_nparray_first_argmax(far_array)
    min_far = far_array[min_far_idx]
    max_far = far_array[max_far_idx]

    f1 = interpolate.interp1d(far_array[min_far_idx:max_far_idx],
                              val_array[min_far_idx:max_far_idx])
    f2 = interpolate.interp1d(far_array[min_far_idx:max_far_idx],
                              thresholds[min_far_idx:max_far_idx],
                              kind='slinear')

    outputs = []

    for idx, far_t in enumerate(far_targets):
        if far_t >= max_far:
            val = val_array[max_far_idx]
            thresh = thresholds[max_far_idx]
        elif far_t <= min_far:
            val = val_array[min_far_idx]
            thresh = thresholds[min_far_idx]
        else:
            val = f1(far_t)
            thresh = f2(far_t)

        t_dict = {
            'FAR': far_t,
            'VAL': val.tolist(),
            'threshold': thresh.tolist()
        }

        outputs.append(t_dict)

    return outputs


def calc_roc_kfolds(thresholds,
                    dist,
                    actual_issame,
                    distance='cosine',
                    nrof_folds=10):
    """
        ROC evaluation with k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))

    nrof_pairs = min(len(actual_issame), dist.shape[0])

    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for idx, threshold in enumerate(thresholds):
            _, _, acc_train[idx] = calc_accuracy(
                threshold, dist[train_set], actual_issame[train_set])

        best_thresh_idx = np.argmax(acc_train)

        for idx, threshold in enumerate(thresholds):
            tprs[fold_idx, idx], fprs[fold_idx, idx], _ = calc_accuracy(
                threshold, dist[test_set], actual_issame[test_set])

        _, _, accuracy[fold_idx] = calc_accuracy(
            thresholds[best_thresh_idx], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy


def calc_val_kfolds(thresholds,
                    dist,
                    actual_issame,
                    far_target,
                    distance='cosine',
                    nrof_folds=10):
    """
        Calculate VAL(=TPR) at target FARs with k-folds
    """

    if thresholds is None or thresholds == []:
        thresholds = np.sort(dist)

    nrof_thresholds = len(thresholds)

    nrof_pairs = min(len(actual_issame), dist.shape[0])
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(nrof_thresholds)

        for idx, threshold in enumerate(thresholds):
            _, far_train[idx] = calc_val_far(
                threshold, dist[train_set], actual_issame[train_set])

        min_far_idx = sorted_nparray_last_argmin(far_train)
        max_far_idx = sorted_nparray_first_argmax(far_train)
        min_far = far_train[min_far_idx]
        max_far = far_train[max_far_idx]

        if far_target >= max_far:
            threshold = 0.0
        elif far_target <= min_far:
            threshold = thresholds[min_far_idx]
        else:
            f = interpolate.interp1d(far_train[min_far_idx:max_far_idx],
                                     thresholds[min_far_idx:max_far_idx],
                                     kind='slinear')
            threshold = f(far_target)

        val[fold_idx], far[fold_idx] = calc_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    return val_mean, val_std, far_mean

################################################################################


def get_auc(fpr, tpr):
    auc = metrics.auc(fpr, tpr)
    return auc


def get_eer(fpr, tpr):
    x_min = fpr.min()
    x_max = fpr.max()
    f = interpolate.interp1d(fpr, tpr)
    eer = brentq(lambda x: 1. - x - f(x), x_min, x_max)
    return eer


def feature(port, image_file, rect):
    with open(image_file, 'r') as f:
        content = f.read()

    rect = [int(x) if int(x) > 0 else 0 for x in rect]

    uri = "data:application/octet-stream;base64," + \
        base64.b64encode(content)
    r = requests.post("http://127.0.0.1:" + port + "/v1/eval",
                      headers={"Content-Type": "application/json"},
                      data=json.dumps({
                          "data": {
                              "uri": uri,
                              "attribute": {
                                  "pts": [
                                      [rect[0], rect[1]],
                                      [rect[2], rect[1]],
                                      [rect[2], rect[3]],
                                      [rect[0], rect[3]],
                                  ]},
                          },
                      }))
    if r.status_code != 200:
        print image_file, rect
        return False, np.array([0] * 512)
    # assert r.status_code == 200
    return True, np.array(list(struct.unpack('>512f', r.content)))


def features(port, faces, begin, return_dict):
    for i, item in enumerate(faces):
        ok1, feature1 = feature(
            port, os.path.join(image_dir, item[0]), item[1])
        ok2, feature2 = feature(
            port, os.path.join(image_dir, item[2]), item[3])
        if ok1 and ok2:
            return_dict[begin + i] = 1 - np.sum(feature1 * feature2)
            print i, len(faces), 1 - np.sum(feature1 * feature2)
        else:
            print i, len(faces)


def main(port, ones, diffs, image_dir):

    print('\n=======================================')
    print('EVALUATION WITHOUT K-FOLDS')

    dist, issame = [], []

    ones = ones[:]
    diffs = diffs[:]

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    n = 10
    m = len(ones) / n
    for i in range(n):
        items = ones[i*m: len(ones) if i == n - 1 else (i + 1) * m]
        p = Process(target=features, args=(port, items, i*m, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    for ret in return_dict.values():
        dist.append(ret)
        issame.append(1)

    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    n = 10
    m = len(diffs) / n
    for i in range(n):
        items = diffs[i*m: len(diffs) if i == n - 1 else (i + 1) * m]
        p = Process(target=features, args=(port, items, i*m, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    for ret in return_dict.values():
        dist.append(ret)
        issame.append(0)

    dist, issame = np.array(dist), np.array(issame)

    thresholds = None
    tpr, fpr, best_accuracy = calc_roc(thresholds, dist, issame)
    val_far = calc_val(thresholds, dist, issame)

    print('Accuracy: %2.5f @distance_threshold=%2.5f, VAL=%2.5f, FAR=%2.5f'
          % (best_accuracy['accuracy'],
             best_accuracy['threshold'],
             best_accuracy['VAL'],
             best_accuracy['FAR']
             )
          )

    auc = get_auc(fpr, tpr)
    print('Area Under Curve (AUC): %2.5f' % auc)

    eer = get_eer(fpr, tpr)
    print('Equal Error Rate (EER): %2.5f with accuracy=%2.5f' % (eer, 1.0-eer))

    for it in val_far:
        print('Validation rate: %2.5f @ FAR=%2.5f with theshold %2.5f' %
              (it['VAL'], it['FAR'], it['threshold']))

    print('\n=======================================')
    print('EVALUATION WITH K-FOLDS')

    # thresholds = None
    # tpr, fpr, accuracy = calc_roc_kfolds(thresholds, dist, issame)
    # val, val_std, far = calc_val_kfolds(thresholds, dist, issame, 1e-3)

    # print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    # auc = get_auc(fpr, tpr)
    # print('Area Under Curve (AUC): %2.5f' % auc)

    # eer = get_eer(fpr, tpr)
    # print('Equal Error Rate (EER): %2.5f with accuracy=%2.5f' % (eer, 1.0-eer))


if __name__ == '__main__':

    port = sys.argv[1]
    pair_file = sys.argv[2]
    jlist_file = sys.argv[3]
    image_dir = sys.argv[4]

    with open(jlist_file, 'r') as _file:
        faces = json.loads(_file.read())
    facem = {}
    for face in faces:
        if len(face['faces']) == 1:
            facem[face['filename']] = face['faces'][0]['rect']
        else:
            rect = face['faces'][0]['rect']
            max_size = (rect[2] - rect[0]) * (rect[3] - rect[1])
            for i in range(1, len(face['faces'])):
                rect1 = face['faces'][i]['rect']
                if (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) > max_size:
                    rect = rect1
                    max_size = (rect[2] - rect[0]) * (rect[3] - rect[1])
            facem[face['filename']] = rect

    print "Faces: ", len(facem)

    ones = []
    diffs = []
    with open(pair_file, 'r') as _file:
        for line in _file.readlines():
            strs = line.strip().split('\t')
            if len(strs) == 3:
                name1 = '''{}/{}_{:0>4d}.jpg'''.format(strs[0],
                                                       strs[0],
                                                       int(strs[1]))
                name2 = '''{}/{}_{:0>4d}.jpg'''.format(strs[0],
                                                       strs[0],
                                                       int(strs[2]))
                ones.append((name1, facem[name1], name2, facem[name2]))
            elif len(strs) == 4:
                name1 = '''{}/{}_{:0>4d}.jpg'''.format(strs[0],
                                                       strs[0],
                                                       int(strs[1]))
                name2 = '''{}/{}_{:0>4d}.jpg'''.format(strs[2],
                                                       strs[2],
                                                       int(strs[3]))
                diffs.append((name1, facem[name1], name2, facem[name2]))
            else:
                print line

    print "Ones: ", len(ones), "Diffs: ", len(diffs)

    main(port, ones, diffs, image_dir)
