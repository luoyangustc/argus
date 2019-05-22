import sys
sys.path.insert(0, "./src")
import argparse
from cfg import Config
from timer import Timer
from recognizers import TextRecognizer
from other import rank_boxes, post_processing
import cv2
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def init_models():
    text_recognizer = TextRecognizer(Config.RECOG_MODEL_PATH, Config.TEXT_RECOG_ALPHABET)
    return text_recognizer

def text_recog(text_recognizer, text_lines, image_path):
    timer = Timer()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s" % image_path

    im = cv2.imread(image_path)

    timer.tic()
    text_lines = rank_boxes(text_lines)
    predictions = text_recognizer.predict(im, text_lines)
    print "Recognition Time: %f" %timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    return predictions, text_lines

def rank_boxes(boxes):
    def getKey(item):
        return item[1] #sort by y1
    sorted_boxes = sorted(boxes,key=getKey)
    return sorted_boxes

def combine_text(text, text_bboxes, img_type):
    func = post_processing(img_type)
    text_all = func(text, text_bboxes)
    return text_all


def load_text_bboxes(text_path):
    with open(text_path, 'r') as f:
        line = f.read()
        det_result = json.loads(line)
        text_lines = det_result['bboxes']
        img_type = det_result['img_type']
    return text_lines, img_type


def dump_result(text_pred):
    #json_result = json.dumps(text_pred,ensure_ascii=False)
    return text_pred

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
    text_recognizer = init_models()
    if args.image is not None:
        image_path = args.image
        text_bboxes, img_type = load_text_bboxes(image_path+'_bboxes.json')
        predictions, text_bboxes = text_recog(text_recognizer, text_bboxes, image_path)
        text_recog_result = combine_text(predictions, text_bboxes, img_type)
        json_text = dump_result(text_recog_result)
        save_json_in_text(json_text, image_path + '.json')
    else:
        with open(args.imagelist, 'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            text_bboxes, img_type = load_text_bboxes(image_path+'_bboxes.json')
            predictions, text_bboxes = text_recog(text_recognizer, text_bboxes, image_path)
            text_recog_result = combine_text(predictions, text_bboxes, img_type)
            json_text = dump_result(image_path, text_recog_result)
            save_json_in_text(json_text, image_path + '.json')
