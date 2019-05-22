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
from imagenet_eval import imagenet_eval, imagenet_eval_detailed, draw_ap, draw_map
from ds_utils import unique_boxes, filter_small_boxes

imagenet_classes = np.array(['__background__',\
                         'n02672831', 'n02691156', 'n02219486', 'n02419796', 'n07739125', 'n02454379',\
                         'n07718747', 'n02764044', 'n02766320', 'n02769748', 'n07693725', 'n02777292',\
                         'n07753592', 'n02786058', 'n02787622', 'n02799071', 'n02802426', 'n02807133',\
                         'n02815834', 'n02131653', 'n02206856', 'n07720875', 'n02828884', 'n02834778',\
                         'n02840245', 'n01503061', 'n02870880', 'n02879718', 'n02883205', 'n02880940',\
                         'n02892767', 'n07880968', 'n02924116', 'n02274259', 'n02437136', 'n02951585',
                         'n02958343', 'n02970849', 'n02402425', 'n02992211', 'n01784675', 'n03000684',\
                         'n03001627', 'n03017168', 'n03062245', 'n03063338', 'n03085013', 'n03793489',\
                         'n03109150', 'n03128519', 'n03134739', 'n03141823', 'n07718472', 'n03797390',\
                         'n03188531', 'n03196217', 'n03207941', 'n02084071', 'n02121808', 'n02268443',\
                         'n03249569', 'n03255030', 'n03271574', 'n02503517', 'n03314780', 'n07753113',\
                         'n03337140', 'n03991062', 'n03372029', 'n02118333', 'n03394916', 'n01639765',\
                         'n03400231', 'n02510455', 'n01443537', 'n03445777', 'n03445924', 'n07583066',\
                         'n03467517', 'n03483316', 'n03476991', 'n07697100', 'n03481172', 'n02342885',\
                         'n03494278', 'n03495258', 'n03124170', 'n07714571', 'n03513137', 'n02398521',\
                         'n03535780', 'n02374451', 'n07697537', 'n03584254', 'n01990800', 'n01910747',\
                         'n01882714', 'n03633091', 'n02165456', 'n03636649', 'n03642806', 'n07749582',\
                         'n02129165', 'n03676483', 'n01674464', 'n01982650', 'n03710721', 'n03720891',\
                         'n03759954', 'n03761084', 'n03764736', 'n03770439', 'n02484322', 'n03790512',\
                         'n07734744', 'n03804744', 'n03814639', 'n03838899', 'n07747607', 'n02444819',\
                         'n03908618', 'n03908714', 'n03916031', 'n00007846', 'n03928116', 'n07753275',\
                         'n03942813', 'n03950228', 'n07873807', 'n03958227', 'n03961711', 'n07768694',\
                         'n07615774', 'n02346627', 'n03995372', 'n07695742', 'n04004767', 'n04019541',\
                         'n04023962', 'n04026417', 'n02324045', 'n04039381', 'n01495701', 'n02509815',\
                         'n04070727', 'n04074963', 'n04116512', 'n04118538', 'n04118776', 'n04131690',\
                         'n04141076', 'n01770393', 'n04154565', 'n02076196', 'n02411705', 'n04228054',\
                         'n02445715', 'n01944390', 'n01726692', 'n04252077', 'n04252225', 'n04254120',\
                         'n04254680', 'n04256520', 'n04270147', 'n02355227', 'n02317335', 'n04317175',\
                         'n04330267', 'n04332243', 'n07745940', 'n04336792', 'n04356056', 'n04371430',\
                         'n02395003', 'n04376876', 'n04379243', 'n04392985', 'n04409515', 'n01776313',\
                         'n04591157', 'n02129604', 'n04442312', 'n06874185', 'n04468005', 'n04487394',\
                         'n03110669', 'n01662784', 'n03211117', 'n04509417', 'n04517823', 'n04536866',\
                         'n04540053', 'n04542943', 'n04554684', 'n04557648', 'n04530566', 'n02062744',\
                         'n04591713', 'n02391049'])

imagenet_cls_names = np.array(['__background__',\
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
                         'zebra'])

class imagenet(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        # year, image_set = image_set.split('_')
        super(imagenet, self).__init__('imagenet_', image_set, root_path, devkit_path)  # set self.name
        # self.year = year
#	print (devkit_path)
#	print ("devkit")
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'DET')

        self.classes = imagenet_classes 
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
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'DET', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            if self.image_set == "val":
                image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
            elif self.image_set == "train":
                image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
            else:
                image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path,'Data','DET', self.image_set, index + '.JPEG')
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
            for gt in roidb:
                if gt['boxes'].shape[0]==0:
                    print(gt['image'])
                return roidb

        gt_roidb = [self.load_imagenet_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def load_imagenet_annotation(self, index):
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

        filename = os.path.join(self.data_path, 'Annotations','DET',self.image_set, index + '.xml')
#	print (filename)
        tree = ET.parse(filename)
    #print(tree)
        objs = tree.findall('object')
#        if not self.config['use_diff']:
 #           non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
 #           objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) 
            y1 = float(bbox.find('ymin').text) 
            x2 = float(bbox.find('xmax').text)
            if x2 == size[1]:
                print ("label xmax reach the image width")
                x2 = x2 - 1
            y2 = float(bbox.find('ymax').text)
            if y2 == size[0]:
                print ("label ymax reach the image height")
                y2 = y2 - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        #'gt_overlaps': overlaps,
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

    def evaluate_detections(self, detections, detailed=False):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'ImageNet')
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'ImageNet' , 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()
        if detailed:
            self.do_python_eval_detailed()

    def boxvoting(self, detections_list):
        all_boxes = [[[] for _ in xrange(self.num_images)]
                 for _ in xrange(self.num_classes)]

        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            for im_ind, index in enumerate(self.image_set_index):
                dets = []
                #for i in range(detections_list.shape[0]):
#      dets.append() =
                #if len(dets) == 0:
                    #continue
                        # the VOCdevkit expects 1-based indices
                    #for k in range(dets.shape[0]):
#           f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                #    format(index, dets[k, -1],
                #       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections_merge(self, detections_list):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        if detections_list.shape[0] <=1:
            detections = detections_list
        else:
            detections = self.boxvoting(detections_list)
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'ImageNet')
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'ImageNet' , 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()


    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = os.path.join(self.devkit_path, 'results', 'ImageNet', 'Main')
        #comp_id = self.config['comp_id']
        #filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        filename = '_det_' + self.image_set + '_{:s}.txt'
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
        annopath = os.path.join(self.data_path, 'Annotations',"DET",self.image_set, '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'DET', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')
        aps = []
        ars = []
        nobs = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True # if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap, ar, npos = imagenet_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            ars += [ar]
            nobs += [npos]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        #self.ap = aps
        draw_ap(aps, ars, nobs, imagenet_cls_names[1:], range_name='all', tag='map={:.4f}'.format(np.mean(aps)))


    def save_ap(self,path = "saveap.txt"):
        aps=[]
        with open(path,"w") as f:
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                filename = self.get_result_file_template().format(cls)
                rec, prec, ap = imagenet_eval(filename, self.annopath, self.imageset_file, cls, self.annocache,
                                     ovthresh=0.5, use_07_metric=True)
                aps += [ap]
                f.write('AP for {} = {:.4f}'.format(cls, ap))
            f.write('Mean AP = {:.4f}'.format(np.mean(aps)))


    def do_python_eval_detailed(self):
        """
        python evaluation wrapper
        :return: None
        """
        annopath = os.path.join(self.data_path, 'Annotations',"DET",self.image_set, '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'DET', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        # The PASCAL VOC metric changed in 2010
        use_07_metric = True  # if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))

        log_aspect_ratio_names = ['<-3', '-3~-1.5', '-1.5~-0.5', '-0.5~0.5', '0.5~1.5', '1.5~3', '>3']
        log_aspect_ratio_ranges = [[-1e5, -3], [-3, -1.5], [-1.5, -0.5], [-0.5, 0.5],
                                   [0.5, 1.5], [1.5, 3], [3, 1e5]]
        log_area_names = ['<13', '13~15', '15~17', '17~19', '>19']
        log_area_ranges = [[0, 13], [13, 15], [15, 17], [17, 19], [19, 1e5]]

        # log_aspect_ratio_ranges, log_aspect_ratio_names = self.get_ranges(start = -3, end = 3, step = 0.2)
        # log_area_ranges, log_area_names = self.get_ranges(start = 8, end = 19, step = 0.2)

        log_area_map = []
        nobs_in_range = []
        for range_id, log_area_range in enumerate(log_area_ranges):
            aps = []
            ars = []
            nobs = []
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                filename = self.get_result_file_template().format(cls)
                rec, prec, ap, ar, npos = imagenet_eval_detailed(filename, annopath, imageset_file, cls, annocache,
                                                            ovthresh=0.5, use_07_metric=use_07_metric, tag='area',
                                                            log_area_range=log_area_range)
                aps += [ap]
                ars += [ar]
                nobs += [npos]
                print('AP for {} = {:.4f} in log area range: [{},{}]'
                      .format(imagenet_cls_names[cls_ind], ap, log_area_range[0], log_area_range[1]))
            draw_ap(aps, ars, nobs, imagenet_cls_names[1:], log_area_names[range_id], tag='log_area')
            nobs_in_range += [np.sum(nobs)]

            # map = np.sum(np.array(aps) * np.array(nobs)) / np.maximum(np.sum(nobs), np.finfo(np.float64).eps)
            map = np.mean(aps)
            print('Mean AP = {:.4f} in log area range: [{},{}]'
                  .format(map, log_area_range[0], log_area_range[1]))
            log_area_map += [map]
        draw_map(log_area_map, log_area_names, nobs_in_range, tag='log_area')
        print('map for area all:{}, num of gt:{}'.format(log_area_map, nobs_in_range))

        log_aspect_ratio_map = []
        nobs_in_range = []
        for range_id, log_aspect_ratio_range in enumerate(log_aspect_ratio_ranges):
            aps = []
            ars = []
            nobs = []
            for cls_ind, cls in enumerate(self.classes):
                if cls == '__background__':
                    continue
                filename = self.get_result_file_template().format(cls)
                rec, prec, ap, ar, npos = imagenet_eval_detailed(filename, annopath, imageset_file, cls, annocache,
                                                            ovthresh=0.5, use_07_metric=use_07_metric,
                                                            tag='aspect ratio',
                                                            log_aspect_ratio_range=log_aspect_ratio_range)
                aps += [ap]
                ars += [ar]
                nobs += [npos]
                print('AP for {} = {:.4f} in log aspect ratio range: [{},{}]'
                      .format(imagenet_cls_names[cls_ind], ap, log_aspect_ratio_range[0], log_aspect_ratio_range[1]))
            draw_ap(aps, ars, nobs, imagenet_cls_names[1:], log_aspect_ratio_names[range_id], tag='log_aspect_ratio')
            nobs_in_range += [np.sum(nobs)]
            print('nobs in this range:{},sum:{}'.format(nobs, np.sum(nobs)))
            # map = np.sum(np.array(aps) * np.array(nobs)) / np.maximum(np.sum(nobs), np.finfo(np.float64).eps)
            map = np.mean(aps)
            print('Mean AP = {:.4f} in log aspect ratio range: [{},{}]'
                  .format(map, log_aspect_ratio_range[0], log_aspect_ratio_range[1]))
            log_aspect_ratio_map += [map]
        draw_map(log_aspect_ratio_map, log_aspect_ratio_names, nobs_in_range, tag='log_aspect_ratio')
        print('map for ratio all:{}, num of gt:{}'.format(log_aspect_ratio_map,nobs_in_range))


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
