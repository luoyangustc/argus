from __future__ import print_function
import numpy as np
import os
import math
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
    parse imagenet record into a dictionary
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
        # obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        x1 = int(float(bbox.find('xmin').text)) * im_scale
        y1 = int(float(bbox.find('ymin').text)) * im_scale
        x2 = int(float(bbox.find('xmax').text)) * im_scale
        y2 = int(float(bbox.find('ymax').text)) * im_scale
        width = x2 - x1
        height = y2 - y1


        obj_dict['bbox'] = [x1, y1, x2, y2]
        obj_dict['bbox_area'] = width * height
        obj_dict['bbox_aspect_ratio'] = (float(height)) / (width)
        obj_dict['im_scale'] = im_scale
        objects.append(obj_dict)
    return objects


def get_imagenet_recs(imageset_file, annopath, annocache, ovthresh=0.5):

    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    #image_filenames = [x.strip() for x in lines]
    image_filenames  = [x.strip().split(' ')[0] for x in lines]

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
    # print(recs)
    return recs

def get_bboxs_area(recs):
    areas = []
    for idx, objects in enumerate(recs.values()):
        for obj in objects:
            areas.append(obj['bbox_area'])
        if idx % 100 == 0:
            print('getting bbox area for {:d}/{:d}'.format(idx + 1, len(recs)))

    areas = np.array(areas)
    print("areas: min:{}, max:{}, num:{}".format(np.amax(areas), np.amin(areas),np.shape(areas)))

    return areas

def get_bboxs_aspect_ratio(recs):
    aspect_ratios= []
    for idx, objects in enumerate(recs.values()):
        for obj in objects:
            aspect_ratios.append(obj['bbox_aspect_ratio'])
        if idx % 100 == 0:
            print('getting bbox aspect ratio for {:d}/{:d}'.format(idx + 1, len(recs)))

    aspect_ratios = np.array(aspect_ratios)
    print("aspect ratios: min:{}, max:{}, num:{}".format(np.amax(aspect_ratios), np.amin(aspect_ratios), np.shape(aspect_ratios)))

    return aspect_ratios


def draw_histogram(data, bin_stride, image_save_path, tag):
    print("draw histogram:{}".format(tag))

    if tag == 'aspect ratio':
        minvalue = 0 # np.amin(data)
        maxvalue = 8 # np.amax(data)
        num_bins = np.arange(minvalue-bin_stride, maxvalue+bin_stride, bin_stride) 
    elif tag == "log data":
        data = np.log2(data)
        minvalue = np.amin(data)
        maxvalue = np.amax(data)
        num_bins = np.arange(minvalue - bin_stride, maxvalue + bin_stride, bin_stride)
        print("log min:{}. log max:{}".format(minvalue, maxvalue))
    else:
        minvalue = np.amin(data)
        maxvalue = np.amax(data)
        num_bins = np.arange(minvalue - bin_stride, maxvalue + bin_stride, bin_stride)

    fig = plt.figure()
    weights =np.ones_like(data)/len(data)
    plt.hist(data, num_bins, weights=weights, alpha=0.5)
    plt.title('bbox {} statistics'.format(tag))
    plt.xlabel('bbox {} (bin size = {})'.format(tag, bin_stride))
    plt.ylabel('percentage')

    plt.savefig(image_save_path)
    plt.close(fig)


if __name__ == '__main__':
    # analyze imagenet
    db = 'imagenet'
    root_path = '/disk2/data/ILSVRC2017/ILSVRC'
    imageset = 'train'
    imageset_file = os.path.join(root_path, 'ImageSets', 'DET', imageset + '.txt')
    annopath = os.path.join(root_path, 'Annotations', 'DET', imageset, '{0!s}.xml')
    annocache = 'imagenet_2017_' + imageset + '_annotations.pkl'

    """
    # analyze voc
    db = 'voc'
    root_path = '/disk2/data/VOCdevkit/VOCdevkit/VOC2007'
    imageset = 'train_val'
    imageset_file = os.path.join(root_path, 'ImageSets', 'Main', imageset + '.txt')
    annopath = os.path.join(root_path, 'Annotations', '{0!s}.xml')
    annocache = 'VOC_2007_' + imageset + 'annotations.pkl'
    """

    recs = get_imagenet_recs(imageset_file, annopath, annocache, ovthresh=0.5)

    area = get_bboxs_area(recs)
    draw_histogram(area, 10000, db + '_'+ imageset + '_' + 'area.png', 'area')
    draw_histogram(area, 0.2, db + '_'+ imageset + '_' + 'log_area.png', 'log data')

    aspect_ratios = get_bboxs_aspect_ratio(recs)
    draw_histogram(aspect_ratios, 0.2, db + '_'+ imageset + '_' + 'aspect_ratio.png', 'aspect ratio')
    draw_histogram(aspect_ratios, 0.2, db + '_' + imageset + '_' + 'aspect_ratio.png', 'log data')
