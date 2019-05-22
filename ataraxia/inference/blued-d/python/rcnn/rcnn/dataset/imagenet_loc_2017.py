from __future__ import print_function
import cPickle
import cv2
import os
import numpy as np

from imdb import IMDB


class imagenet_loc_2017(IMDB):
    def __init__(self, image_set, root_path, data_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param data_path: data and results
        :return: imdb object
        """
        super(imagenet_loc_2017, self).__init__(
            'imagenet_loc_2017_', image_set, root_path, data_path)
        print('initialize imagenet loc 2017')
        self.root_path = root_path
        self.data_path = os.path.join(root_path, data_path)
        self.class_to_index = self._calc_class_to_index()
        self.classes = np.array(self.class_to_index.keys())
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print('num_images', self.num_images)

        self.config = {'comp_id': 'comp4', 'use_diff': False, 'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        print('load image index set')
        image_set_index_file = self._get_index_file()
        assert os.path.exists(
            image_set_index_file), 'Path does not exist: {}'.format(
                image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.split(' ')[0].strip() for x in f.readlines()]
        return image_set_index

    def _get_index_file(self):
        return os.path.join(self.root_path, self.image_set + '_loc.txt')


    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'Data', 'CLS-LOC',
                                  self.image_set, index + '.JPEG')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(
            image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print('get gt roidb')
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            for gt in roidb:
                if gt['boxes'].shape[0] == 0:
                    print(gt['image'])
            return roidb

        gt_roidb = [
            self.load_imagenet_annotation(index)
            for index in self.image_set_index
        ]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _calc_class_to_index(self):
        class_to_index = {}
        with open(os.path.join(self.data_path, 'devkit', 'data', 'map_clsloc.txt')) as f:
            for l in f.readlines():
                elem = l.strip().split(' ')
                class_to_index[elem[0]] = int(elem[1])

        return class_to_index

    def load_imagenet_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print('load imagenet annotation', index)
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        size = cv2.imread(roi_rec['image']).shape
        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]

        filename = os.path.join(self.data_path, 'Annotations', 'CLS-LOC',
                                self.image_set, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes + 1), dtype=np.float32)

        class_to_index = self.class_to_index
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            if x2 == size[1]:
                print("label xmax reach the image width")
                x2 = x2 - 1
            y2 = float(bbox.find('ymax').text)
            if y2 == size[0]:
                print("label ymax reach the image height")
                y2 = y2 - 1
            class_name = obj.find('name').text.lower().strip()
            if class_name not in class_to_index:
                print(class_name, 'not in', class_to_index)
            cls = class_to_index[class_name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'max_classes': overlaps.argmax(axis=1),
            'max_overlaps': overlaps.max(axis=1),
            'flipped': False
        })
        return roi_rec
