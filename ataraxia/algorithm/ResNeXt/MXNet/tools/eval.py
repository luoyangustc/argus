#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import json
import matplotlib.pyplot as plt
import itertools
import numpy as np
import json
import os
import docopt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def eval(pred, gt, classes):

    def _max(lst):
        m = max(lst)
        return [i for i, j in enumerate(lst) if j == m][0]

    pred_max = [_max(a) for a in pred]
    p = precision_score(gt, pred_max, average=None)
    r = recall_score(gt, pred_max, average=None)
    acc = accuracy_score(gt, pred_max)
    print('accuracy: {}'.format(acc))

    for i in range(len(classes)):
        gt_score = []
        pred_score = []
        for j in range(len(pred)):
            pred_score.append(pred[j][i])
            gt_score.append(1.0 if gt[j] == i else 0.0)
        ap = average_precision_score(gt_score, pred_score)
        print('{} precision: {}'.format(classes[i], p[i]))
        print('{} recall: {}'.format(classes[i], r[i]))
        print('{} ap: {}'.format(classes[i], ap))

    print('Top-1 error ', 1 - acc)
    cm = confusion_matrix(gt, pred_max, labels=np.arange(len(classes)))
    plot_confusion_matrix(cm, classes)


def eval_pulp(output_file, gt_file):
    classes = ['pulp', 'sexy', 'normal']
    pred_m = {}
    with open(output_file, 'r') as f:
        d = json.load(f)
        for k, v in d.items():
            pred_m[k] = [float(a) for a in v['Confidence']]
    gt = []
    pred = []
    with open(gt_file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            name = os.path.basename(data['url'])
            if name not in pred_m:
                continue
            clas = data['label'][0]['data'][0]['class']
            gt.append([i for i, j in enumerate(classes) if j == clas][0])
            pred.append(pred_m[name])
    eval(pred, gt, classes)


if __name__ == "__main__":
    import sys
    eval_pulp(sys.argv[1], sys.argv[2])
