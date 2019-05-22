import sys
sys.path.insert(0, "./src")

import argparse
from cfg import Config
from other import draw_boxes, resize_im, refine_boxes, calc_area_ratio, CaffeModel
import cv2, caffe
from detectors import TextProposalDetector, TextDetector
from utils.timer import Timer
import json

def init_models():
    if Config.PLATFORM == "GPU":
        caffe.set_mode_gpu()
        caffe.set_device(Config.TEST_GPU_ID)
    else:
        caffe.set_mode_cpu()

    # initialize the detectors
    text_proposals_detector=TextProposalDetector(CaffeModel(Config.DET_NET_DEF_FILE, Config.DET_MODEL_PATH))
    text_detector=TextDetector(text_proposals_detector)

    return text_detector

def text_detect(text_detector, image_path, img_type):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%image_path
    if img_type == "others":
        return [],0

    im=cv2.imread(image_path)
    im_small, f=resize_im(im, Config.SCALE, Config.MAX_SCALE)

    timer = Timer()
    timer.tic()
    text_lines = text_detector.detect(im_small)
    text_lines = text_lines / f # project back to size of original image
    text_lines = refine_boxes(im, text_lines, expand_pixel_len = Config.DILATE_PIXEL,
                              pixel_blank = Config.BREATH_PIXEL, binary_thresh=Config.BINARY_THRESH)
    text_area_ratio = calc_area_ratio(text_lines, im.shape)
    print "Number of the detected text lines: %s" % len(text_lines)
    print "Detection Time: %f" % timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if Config.DEBUG_SAVE_BOX_IMG:
        im_with_text_lines = draw_boxes(im, text_lines, is_display=False, caption=image_path, wait=False)
        if im_with_text_lines is not None:
            cv2.imwrite(image_path+'_boxes.jpg', im_with_text_lines)

    return text_lines, text_area_ratio

def dump_result(image_path, text_lines, text_area_ratio, img_type):
    text_detect_result = dict()
    text_detect_result['img'] = image_path
    text_detect_result['bboxes'] = text_lines
    text_detect_result['area_ratio'] = text_area_ratio
    text_detect_result['img_type'] = img_type

    json_result = json.dumps(text_detect_result)
    return json_result

def load_text_bboxes(text_path):
    with open(text_path, 'r') as f:
        line = f.read()
        cls_result = json.loads(line)
        img_type = cls_result['img_type']
    return img_type

def save_json_in_text(json_result, text_path):
    with open(text_path,'w') as f:
        f.write(json_result)

def parse_args():
    parser = argparse.ArgumentParser(description='Test OCR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    text_detector = init_models()
    if args.image is not None:
        image_path = args.image
        img_type = load_text_bboxes(image_path+'_cls.json')
        text_bboxes, text_area_ratio = text_detect(text_detector, image_path, img_type)
        json_result = dump_result(image_path, text_bboxes, text_area_ratio, img_type)
        print json_result
        if text_bboxes is not None:
            save_json_in_text(json_result, image_path+'_bboxes.json')
    else:
        with open(args.imagelist,'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            img_type = load_text_bboxes(image_path + '_cls.json')
            text_bboxes, text_area_ratio = text_detect(text_detector, image_path, img_type)
            json_result = dump_result(image_path, text_bboxes, text_area_ratio, img_type)
            print json_result
            if text_bboxes is not None:
                save_json_in_text(json_result, image_path+'_bboxes.json')
