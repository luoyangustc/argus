from cfg import Config
import io

def rank_boxes(boxes):
    def getKey(item):
        return item[1] #sort by y1
    sorted_boxes = sorted(boxes,key=getKey)
    return sorted_boxes

def post_processing(img_type):
    if img_type == "blog":
        return _blog_post_processing
    elif img_type == "wechat":
        return _wechat_post_processing
    else:
        return _general_post_processing

def load_stop_punctuation(path="stop_punctuation.utf8"):
    stop_punctuation = set()
    with io.open(path, 'r', encoding='utf8') as f:
        for l in f:
            l = l.strip()
            stop_punctuation.add(l)
    return stop_punctuation

def _blog_post_processing(text, text_bboxes):
    result = {}
    result["texts"]=[]
    result["bboxes"]=[]
    for idx, text_bbox in enumerate(text_bboxes):
        temp_res = []
        temp_point1 = []
        temp_point2 = []
        temp_point3 = []
        temp_point4 = []
        temp_point1.append(text_bbox[0])
        temp_point1.append(text_bbox[1])
        temp_point2.append(text_bbox[2])
        temp_point2.append(text_bbox[1])
        temp_point3.append(text_bbox[2])
        temp_point3.append(text_bbox[3])
        temp_point4.append(text_bbox[0])
        temp_point4.append(text_bbox[3])
        temp_res.append(temp_point1)
        temp_res.append(temp_point2)
        temp_res.append(temp_point3)
        temp_res.append(temp_point4)
        result["bboxes"].append(temp_res)
        result["texts"].append(text[idx])
    return result

def _wechat_post_processing(text, text_bboxes):
    return _general_post_processing(text, text_bboxes)

def _general_post_processing(text, text_bboxes):
    result = {}
    result["texts"] = []
    result["bboxes"] = []
    for idx, text_bbox in enumerate(text_bboxes):
        temp_res = []
        temp_point1 = []
        temp_point2 = []
        temp_point3 = []
        temp_point4 = []
        temp_point1.append(text_bbox[0])
        temp_point1.append(text_bbox[1])
        temp_point2.append(text_bbox[2])
        temp_point2.append(text_bbox[1])
        temp_point3.append(text_bbox[2])
        temp_point3.append(text_bbox[3])
        temp_point4.append(text_bbox[0])
        temp_point4.append(text_bbox[3])
        temp_res.append(temp_point1)
        temp_res.append(temp_point2)
        temp_res.append(temp_point3)
        temp_res.append(temp_point4)
        result["bboxes"].append(temp_res)
        result["texts"].append(text[idx])
    return result
