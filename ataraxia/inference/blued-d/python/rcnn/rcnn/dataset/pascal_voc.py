"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

from __future__ import print_function
import cPickle
import cv2
import os
import numpy as np

from imdb import IMDB
from pascal_voc_eval import voc_eval, voc_eval_detailed, draw_ap, draw_map, draw_pr_curve
from ds_utils import unique_boxes, filter_small_boxes


class PascalVOC(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        year, image_set = image_set.split('_')
        super(PascalVOC, self).__init__('voc_' + year, image_set, root_path, devkit_path)  # set self.name
        self.year = year
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'VOC' + year)

        self.classes = ['__background__',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print('num_images', self.num_images)

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self.load_pascal_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        size = cv2.imread(roi_rec['image']).shape
        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]

        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    def load_selective_search_roidb(self, gt_roidb):
        """
        turn selective search proposals into selective search roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import scipy.io
        matfile = os.path.join(self.root_path, 'selective_search_data', self.name + '.mat')
        assert os.path.exists(matfile), 'selective search data does not exist: {}'.format(matfile)
        raw_data = scipy.io.loadmat(matfile)['boxes'].ravel()  # original was dict ['images', 'boxes']

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1  # pascal voc dataset starts from 1.
            keep = unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_roidb(self, gt_roidb, append_gt=False):
        """
        get selective search roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of selective search
        """
        cache_file = os.path.join(self.cache_path, self.name + '_ss_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if append_gt:
            print('appending ground truth annotations')
            ss_roidb = self.load_selective_search_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self.load_selective_search_roidb(gt_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()
        self.do_python_eval_detailed()

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: None
        """
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')
        aps = []
        ars = []
        nobs = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap, ar, npos = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            ars += [ar]
            nobs += [npos]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        draw_ap(aps, ars, nobs, self.classes[1:], range_name='all', tag='map = {:.4f}'.format(np.mean(aps)))


    def do_python_eval_detailed(self):
        """
        python evaluation wrapper
        :return: None
        """
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        # The PASCAL VOC metric changed in 2010
        use_07_metric = True # if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))


        log_aspect_ratio_names = ['<-3','-3~-1.5', '-1.5~-0.5','-0.5~0.5','0.5~1.5','1.5~3','>3']
        log_aspect_ratio_ranges = [[-1e5, -3] ,[-3, -1.5], [-1.5, -0.5], [-0.5, 0.5],
                                   [0.5, 1.5], [1.5, 3], [3, 1e5]]
        log_area_names = ['<13', '13~15', '15~17', '17~19', '>19']
        log_area_ranges = [[0, 13], [13, 15], [15, 17], [17, 19], [19, 1e5]]

        # log_aspect_ratio_ranges, log_aspect_ratio_names = self.get_ranges(start = -3, end = 3, step = 0.2)
        # log_area_ranges, log_area_names = self.get_ranges(start = 8, end = 19, step = 0.2)

        log_area_map = []
        for range_id, log_area_range in enumerate(log_area_ranges):
            aps = []
            ars = []
            nobs = []
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                filename = self.get_result_file_template().format(cls)
                rec, prec, ap, ar, npos = voc_eval_detailed(filename, annopath, imageset_file, cls, annocache,
                                         ovthresh=0.5, use_07_metric=use_07_metric, tag='area', log_area_range = log_area_range)
                draw_pr_curve(rec, prec, 'log_area', log_area_names[range_id], cls)

                aps += [ap]
                ars += [ar]
                nobs += [npos]
                print('AP for {} = {:.4f} in log area range: [{},{}]'
                      .format(cls, ap, log_area_range[0], log_area_range[1]))
            draw_ap(aps, ars, nobs, self.classes[1:], log_area_names[range_id],tag='log_area')

            map = np.mean(aps)
            print('Mean AP = {:.4f} in log area range: [{},{}]'
                  .format(map, log_area_range[0], log_area_range[1]))
            log_area_map += [map]
        draw_map(log_area_map, log_area_names, tag='log_area')
        print('map for area all:', log_area_map)


        log_aspect_ratio_map = []
        for range_id, log_aspect_ratio_range in enumerate(log_aspect_ratio_ranges):
            aps = []
            ars = []
            nobs = []
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                filename = self.get_result_file_template().format(cls)
                rec, prec, ap, ar, npos = voc_eval_detailed(filename, annopath, imageset_file, cls, annocache,
                                         ovthresh=0.5, use_07_metric=use_07_metric, tag='aspect ratio',
                                         log_aspect_ratio_range= log_aspect_ratio_range)
                aps += [ap]
                ars += [ar]
                nobs += [npos]
                print('AP for {} = {:.4f} in log aspect ratio range: [{},{}]'
                      .format(cls, ap, log_aspect_ratio_range[0], log_aspect_ratio_range[1]))
            draw_ap(aps, ars, nobs, self.classes[1:], log_aspect_ratio_names[range_id], tag='log_aspect_ratio')

            # map = np.sum(np.array(aps) * np.array(nobs)) / np.maximum(np.sum(nobs), np.finfo(np.float64).eps)
            map = np.mean(aps)
            print('Mean AP = {:.4f} in log aspect ratio range: [{},{}]'
                  .format(map, log_aspect_ratio_range[0], log_aspect_ratio_range[1]))
            log_aspect_ratio_map += [map]
        draw_map(log_aspect_ratio_map, log_aspect_ratio_names, tag='log_aspect_ratio')
        print('map for ratio all:', log_aspect_ratio_map)

    def get_ranges(self, start, end, step):
        v = np.arange(start, end, step)
        v = np.insert(v, 0, -1e5)
        v = np.append(v, 1e5)

        ranges = []
        range_names = []
        for idx in range(len(v) - 1):
            range_start = v[idx]
            range_end = v[idx + 1]
            # if start/end is very close to zero, set it to zero
            if range_start > -1e-10 and range_start < 1e-10:
                range_start = 0
            if range_end > -1e-10 and range_end < 1e-10:
                range_end = 0
            ranges.append([range_start, range_end])
            # set names of first and last range
            if idx == 0:
                name = '<' + str(range_end)
            elif idx == len(v) - 2:
                name = '>' + str(range_start)
            else:
                name = str(range_start) + '~' + str(range_end)

            range_names.append(name)

        print(range_names)
        print(ranges)

        return ranges, range_names


