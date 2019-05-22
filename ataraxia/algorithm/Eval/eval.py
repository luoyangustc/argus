import pandas as pd
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

dic_pred = {}
dic_pred_score = {}
dic_gt = {}
dic_gt_score = {}
y_gt = []
y_pred=[]
y_gt_score = []
y_score = []
pulp_score_list = []
sexy_score_list = []
normal_score_list = []
pulp_gt_list = []
sexy_gt_list = []
normal_gt_list = []

with open("output.json") as f1 ,open("groundtruth.json") as f2:
    lines2 = f2.readlines()
    d = json.load(f1)
    for k,v in d.items():
        lst = []
        dic_pred[k] = v['Top-1 Index'][0]
        lst.append(float(v["Confidence"][0]))
        lst.append(float(v["Confidence"][1]))
        lst.append(float(v["Confidence"][2]))
        dic_pred_score[k] = lst

    for line in lines2:
        lst = []
        basename = os.path.basename(json.loads(line.strip())['url'])
        classes =  json.loads(line.strip())['label'][0]['data'][0]["class"]
        if classes == "sexy":
            index = 1
            lst = [0,1,0]
        elif classes == "pulp":
            index = 0
            lst = [1,0,0]
        else:
            index = 2
            lst = [0,0,1]
        dic_gt[basename] = index
        dic_gt_score[basename] = lst


for k in dic_pred:
    if k in dic_gt:
        y_gt.append(dic_gt[k])
        y_pred.append(dic_pred[k])

        pulp_score_list.append(dic_pred_score[k][0])
        sexy_score_list.append(dic_pred_score[k][1])
        normal_score_list.append(dic_pred_score[k][2])

        pulp_gt_list.append(dic_gt_score[k][0])
        sexy_gt_list.append(dic_gt_score[k][1])
        normal_gt_list.append(dic_gt_score[k][2])


y_score.append(pulp_score_list)
y_score.append(sexy_score_list)
y_score.append(normal_score_list)

y_gt_score.append(pulp_gt_list)
y_gt_score.append(sexy_gt_list)
y_gt_score.append(normal_gt_list)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None):
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

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classes = ['pulp','sexy','normal']
cm = confusion_matrix(y_gt, y_pred, labels=np.arange(len(classes)))
p = precision_score(y_gt, y_pred, average=None)
r = recall_score(y_gt, y_pred, average=None)
# ap = average_precision_score(y_gt,y_pred)
acc = accuracy_score(y_gt, y_pred)
print('accuracy:', acc)

for i in range(len(classes)):
    ap = average_precision_score(y_gt_score[i],y_score[i])
    print('%s precision:' % classes[i], p[i])
    print('%s recall:' % classes[i], r[i])
    print('%s ap:'%classes[i],ap)

print('Top-1 error ',1-acc)
plot_confusion_matrix(cm, classes)