import json
import cv2
import numpy as np
import six.moves.cPickle as cPickle  # pylint: disable=no-name-in-module,import-error
from aisdk.framework.base_inference import InferenceReq, BaseInferenceServer
from aisdk.common.image import load_imagev2
from aisdk.common.error import ErrorBase
import csv
import aisdk.proto as pb
from . import const


class InferenceServer(BaseInferenceServer):
    def __init__(self, app_name, cfg):
        super(InferenceServer, self).__init__(app_name)
        self.inference_req = InferenceReq()

        percent_fine = float(cfg['custom_params']['percentage_fine'])

        fine_labels = str(cfg['model_files']["fine_labels.csv"])
        dict_label = parse_label_file(fine_labels)

        det_labelfile = str(cfg['model_files']["det_labels.csv"])
        det_label_dict = parse_label_file(det_labelfile)

        self.cls_model = {
            "percentage_fine": percent_fine,
            "dict_label": dict_label,
        }
        self.det_model = {'label': det_label_dict}
        self.det_label_dict = det_label_dict

    def net_inference(self, request):
        assert isinstance(request, pb.InferenceRequest)
        img = load_imagev2(request.data.body)
        assert img.ndim == 3  # TODO
        img_height, img_width, _ = img.shape
        img_cls = cls_preProcessImage(img)
        img_det = det_preProcessImage(img)
        forward_req = {'img_cls': img_cls, 'img_det': img_det}
        msg = pb.ForwardMsg(
            network_input_buf=cPickle.dumps(
                forward_req, protocol=cPickle.HIGHEST_PROTOCOL),
            reqid=request.reqid)
        msg_out = self.inference_req.inference_msg(msg)
        output = cPickle.loads(msg_out.network_output_buf)
        image_index = json.loads(
            msg_out.meta['data'].decode('utf8'))['image_index']

        cls_result = cls_post_eval(output['output_fine'], image_index,
                                   self.cls_model)
        det_result = det_post_eval(img_height, img_width, output['output_det'],
                                   self.det_label_dict, image_index)
        resp = postProcess(cls_result, det_result)
        return pb.InferenceResponse(code=200, result=json.dumps(resp))


def postProcess(cls_result, det_result):
    return {'classify': cls_result, 'detection': det_result}


def det_post_eval(img_height, img_width, output, label_dict, image_index):
    # output_bbox_list : bbox_count * 7
    output_bbox_list = output['detection_out'][0][0]
    image_result = []
    for i_bbox in output_bbox_list:
        # i_bbox : length == 7 ; 0==image_id,1==class_index,2==score,3==bbox_xmin,4==bbox_ymin,5==bbox_xmax,6==bbox_ymax
        image_id = int(i_bbox[0])
        if image_id != image_index:
            continue
        h = img_height
        w = img_width
        class_index = int(i_bbox[1])
        if class_index < 1:  # background index == 0 , refinedet not output background info ,so the line not used
            continue
        score = float(i_bbox[2])
        if score < float(label_dict['threshold'][class_index]):
            continue
        name = label_dict['class'][class_index]
        bbox_dict = dict()
        bbox_dict['index'] = class_index
        bbox_dict['score'] = score
        bbox_dict['class'] = name
        bbox = i_bbox[3:7] * np.array([w, h, w, h])
        bbox_dict['pts'] = []
        xmin = int(bbox[0]) if int(bbox[0]) > 0 else 0
        ymin = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xmax = int(bbox[2]) if int(bbox[2]) < w else w
        ymax = int(bbox[3]) if int(bbox[3]) < h else h
        bbox_dict['pts'].append([xmin, ymin])
        bbox_dict['pts'].append([xmax, ymin])
        bbox_dict['pts'].append([xmax, ymax])
        bbox_dict['pts'].append([xmin, ymax])
        image_result.append(bbox_dict)
    return image_result


def parse_label_file(labelFile=None):
    label_dict = dict()
    with open(labelFile, 'r') as f:
        line_list = [i.strip() for i in f.readlines() if i]
        # Add by Riheng
        keyList = line_list[0].split(',')  # index,class,threshold
        for key in keyList[1:]:
            label_dict[key] = dict()
        for i_line in line_list[1:]:
            # Add by Riheng
            i_line_list = i_line.split(',')
            index_value = int(i_line_list[0])
            for colume_index, value in enumerate(i_line_list[1:], 1):
                label_dict[keyList[colume_index]][index_value] = value
    return label_dict


def get_label_map(labelmap):
    map_dict = {}
    with open(labelmap, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            map_dict[int(line[0])] = int(line[1])
    return map_dict


def cls_post_eval(output_fine, image_index, cls_model):
    '''
        parse net output, as numpy.mdarray, to EvalResponse
    '''
    dict_label = cls_model['dict_label']

    output_prob = np.squeeze(output_fine['prob'][image_index])
    index = int(output_prob.argsort()[-1])
    class_name = str(dict_label['class'][index])
    score = float(output_prob[index])
    if score < 0.9:
        class_name = 'normal'
    confidence = {
        'confidences': [{
            "index": index,
            "class": class_name,
            "score": score
        }]
    }
    return confidence


def center_crop(img, crop_size):
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        raise ErrorBase(400, "bad image size")
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy:yy + crop_size, xx:xx + crop_size]


def cls_preProcessImage(img=None):
    img = img.astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img -= np.array([[[103.94, 116.78, 123.68]]])
    img = img * 0.017
    img = center_crop(img, 225)
    img = img.transpose((2, 0, 1))
    return img


def det_preProcessImage(oriImage=None):
    img = cv2.resize(oriImage, (512, 512))
    img = img.astype(np.float32)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img.transpose((2, 0, 1))
    return img


def serve():
    s = InferenceServer(const.app_name, const.cfg)
    s.serve()


if __name__ == '__main__':
    const.flavor.run_inference(serve)
