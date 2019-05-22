"""
given a pascal voc imdb, compute mAP
"""

from __future__ import print_function
import numpy as np
import os
import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def resize(width, height, target_size, max_size):
    if width < height:
        im_size_min = width
        im_size_max = height
    else:
        im_size_min = height
        im_size_max = width

    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    return im_scale


def parse_imagenet_rec(filename, scale=[(600,1000)]):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    size = tree.find('size')
    im_width = int(float(size.find('width').text))
    im_height = int(float(size.find('height').text))
    im_scale = resize(im_width, im_height, scale[0][0], scale[0][1])

    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        #obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        x1 = int(float(bbox.find('xmin').text))
        y1 = int(float(bbox.find('ymin').text))
        x2 = int(float(bbox.find('xmax').text))
        y2 = int(float(bbox.find('ymax').text))
        obj_dict['bbox'] = [x1, y1, x2, y2]
        obj_dict['log_bbox_area'] = np.log2((x2 - x1) * (y2 - y1) * im_scale * im_scale)
        obj_dict['log_bbox_aspect_ratio'] = np.log2((float(y2 - y1)) / (x2 - x1))
        obj_dict['im_scale_xml'] = im_scale
        objects.append(obj_dict)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def imagenet_eval(detpath, annopath, imageset_file, classname, annocache, ovthresh=0.5, use_07_metric=False):
    """
    pascal voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    #image_filenames = [x.strip() for x in lines]
    image_filenames  = [x.split(' ')[0] for x in lines]

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            recs[image_filename] = parse_imagenet_rec(annopath.format(image_filename))
            if ind % 100 == 0:
                print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))
        print('saving annotations cache to {:s}'.format(annocache))
        with open(annocache, 'w') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'r') as f:
            recs = cPickle.load(f)

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in objects])

#        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + len(objects)
        class_recs[image_filename] = {'bbox': bbox,
 #                                    'difficult': difficult,
                                      'det': det}

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    if bbox.shape[0] > 0:
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
        # if not r['difficult'][jmax]:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    ar = voc_ap(prec, rec, use_07_metric)

    return rec, prec, ap, ar, npos


def imagenet_eval_detailed(detpath, annopath, imageset_file, classname,
                      annocache, ovthresh=0.5, use_07_metric=False, tag='area',
                      log_area_range=[0, 1e5], log_aspect_ratio_range=[0, 1e5]):
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    # image_filenames = [x.strip() for x in lines]
    image_filenames = [x.strip().split(' ')[0] for x in lines]

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            recs[image_filename] = parse_imagenet_rec(annopath.format(image_filename))
            if ind % 100 == 0:
                print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))
        print('saving annotations cache to {:s}'.format(annocache))
        with open(annocache, 'w') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'r') as f:
            recs = cPickle.load(f)

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    for image_filename in image_filenames:
        objects_all = [obj for obj in recs[image_filename] if obj['name'] == classname]
        if len(objects_all) > 0:
            im_scale = objects_all[0]['im_scale_xml']
        else:
            im_scale = 1
        if tag == 'area':
            objects = [obj for obj in objects_all
                       if obj['log_bbox_area'] >= log_area_range[0] and obj['log_bbox_area'] < log_area_range[1]]
        if tag == 'aspect ratio':
            objects = [obj for obj in objects_all
                       if obj['log_bbox_aspect_ratio'] >= log_aspect_ratio_range[0] and obj['log_bbox_aspect_ratio'] <
                       log_aspect_ratio_range[1]]

        bbox = np.array([x['bbox'] for x in objects_all])
        log_bbgt_area = np.array([x['log_bbox_area'] for x in objects_all])
        log_bbgt_aspect_ratio = np.array([x['log_bbox_aspect_ratio'] for x in objects_all])

        det = [False] * len(objects_all)  # stand for detected
        npos = npos + len(objects)
        class_recs[image_filename] = {'bbox': bbox,
                                      'log_bbgt_area': log_bbgt_area,
                                      'log_bbgt_aspect_ratio': log_bbgt_aspect_ratio,
                                      'det': det,
                                      'im_scale_xml': im_scale}
    print("num of gt boxes for this cls in this area:", npos)
    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])
    print("num of detections for this cls:", len(image_ids))

    """
    # keep the detection of this area
    keep = []
    if tag == 'area':
        for idx in range(bbox.shape[0]):
            r = class_recs[image_ids[idx]]
            im_scale = r['im_scale_xml']
            x1 = bbox[idx][0] / im_scale
            y1 = bbox[idx][1] / im_scale
            x2 = bbox[idx][2] / im_scale
            y2 = bbox[idx][3] / im_scale

            log_bbox_area = np.log2((x2 - x1) * (y2 - y1))
            if (log_bbox_area >= log_area_range[0] and log_bbox_area < log_area_range[1]):
                keep.append(idx)

    if tag == 'aspect ratio':
        for idx in range(bbox.shape[0]):
            x1 = bbox[idx][0]
            y1 = bbox[idx][1]
            x2 = bbox[idx][2]
            y2 = bbox[idx][3]
            log_bbox_aspect_ratio = np.log2((float(y2 - y1)) / (x2 - x1))
            if (log_bbox_aspect_ratio >= log_aspect_ratio_range[0] and log_bbox_aspect_ratio < log_aspect_ratio_range[1]):
                keep.append(idx)

    image_ids = [image_ids[idx] for idx in keep]
    confidence = np.array([confidence[idx] for idx in keep])
    bbox = np.array([bbox[idx] for idx in keep])
    """

    # sort by confidence
    if bbox.shape[0] > 0:
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if tag == 'area':
                bbgt_log_area = r['log_bbgt_area'][jmax]
                if (bbgt_log_area >= log_area_range[0] and bbgt_log_area < log_area_range[1]):
                    keep = True
                else:
                    keep = False
            if tag == 'aspect ratio':
                bbgt_log_aspect_ratio = r['log_bbgt_aspect_ratio'][jmax]
                if (bbgt_log_aspect_ratio >= log_aspect_ratio_range[0] and bbgt_log_aspect_ratio <
                    log_aspect_ratio_range[1]):
                    keep = True
                else:
                    keep = False

            if keep:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                continue
        else:
            fp[d] = 1.
            # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #print("recall:{}, precision:{}".format(rec, prec))
    ap = voc_ap(rec, prec, use_07_metric)
    ar = voc_ap(prec, rec, use_07_metric)

    return rec, prec, ap, ar, npos


def draw_pr_curve(rec, prec, tag, range_name, cls):
    plt.figure(1)
    plt.plot(rec, prec)
    title = 'PR_curve_' + tag + '_' + range_name + '_' + cls
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    img_path = title + '.png'
    plt.savefig(img_path)
    plt.close(1)


def draw_ap(cls_aps, cls_ars, cls_gt_boxes, classes, range_name, tag='area'):
    print("drawing ap for:", range_name)
    num_bins = len(classes)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(30,8))

    rects1 = ax1.bar(left=np.arange(num_bins), height=cls_aps, width=0.35, align='center', label='aps')
    ax1.set_title('classes aps and ars in {} [{}]'.format(tag, range_name))
    ax1.set_xticks(np.arange(num_bins))
    ax1.set_xticklabels(classes, rotation=90, fontsize=8)
    ax1.set_ylabel('aps')
    autolabel(rects1, ax1)
    """
    rects2 = ax2.bar(left=np.arange(num_bins), height=cls_ars, width=0.35, align='center', label='aps')
    ax2.set_xticks(np.arange(num_bins))
    ax2.set_xticklabels(classes, rotation=90, fontsize=6)
    ax2.set_ylabel('ars')
    autolabel(rects2, ax2)
    """

    rects3 = ax3.bar(left=np.arange(num_bins), height=cls_gt_boxes, width=0.35, align='center', label='num of gt boxes')
    ax3.set_xlabel('class ids')
    ax3.set_xticks(np.arange(num_bins))
    ax3.set_xticklabels(classes, rotation=90, fontsize=8)
    ax3.set_ylabel('num of gt boxes')
    autolabel(rects3, ax3, floatp=False)
    fig.tight_layout()

    img_save_path = 'ap_' + tag + '_' + range_name + '.png'
    plt.savefig(img_save_path)
    plt.close(fig)


def draw_map(map, ranges, range_gt_boxes, tag='area'):
    print("drawing map")
    print(map,range_gt_boxes)
    num_bins = len(ranges)
    fig, (ax1, ax2) = plt.subplots(2, 1)

    rects1 = ax1.bar(left=np.arange(num_bins), height=map, width=0.35, align='center')
    ax1.set_title('maps for {}'.format(tag))
    ax1.set_xticks(np.arange(num_bins))
    ax1.set_xticklabels(ranges, rotation=90, fontsize=6)
    ax1.set_ylabel('maps')
    autolabel(rects1, ax1)

    rects2 = ax2.bar(left=np.arange(num_bins), height=range_gt_boxes, width=0.35, align='center', label='num of gt boxes')
    ax2.set_xlabel('ranges')
    ax2.set_xticks(np.arange(num_bins))
    ax2.set_xticklabels(ranges, rotation=90, fontsize=6)
    ax2.set_ylabel('num of gt boxes')
    autolabel(rects2, ax2, floatp=False)
    fig.tight_layout()

    img_save_path = 'map_' + tag + '.png'
    plt.savefig(img_save_path)
    plt.close(fig)


def autolabel(rects, ax, floatp=True):
    for rect in rects:
        height = rect.get_height()
        if floatp:
            ax.text(rect.get_x(), 1.03 * height, "%.2f" % float(height), fontsize=6)
        else:
            ax.text(rect.get_x(), 1.03 * height, "%s" % int(height), fontsize=6)
