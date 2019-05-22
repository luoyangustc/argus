import json
import numpy as np
import six.moves.cPickle as cPickle  # pylint: disable=no-name-in-module,import-error
import os
import os.path
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

from aisdk.framework.base_inference import InferenceReq, BaseInferenceServer
from aisdk.common.image import load_imagev2
from aisdk.common.error import ErrorBase
import aisdk.common.other
import aisdk.proto as pb
from . import const

from rfcn.config.config import config, update_config
from lib.utils.image import resize, transform
from nms.nms import py_nms_wrapper


def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    im_scale: float number
    """
    SHORT_SIDE = config.SCALES[0][0]
    LONG_SIDE = config.SCALES[0][1]
    PIXEL_MEANS = config.network.PIXEL_MEANS

    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_array = im_array.astype(np.float32)
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]],
                       dtype=np.float32)
    data_shapes = [[('data', im_array.shape), ('im_info', im_info.shape)]]
    return {
        'data_shapes': data_shapes,
        'im_scale': [im_scale],
        'im_array': im_array,
        'im_info': im_info
    }


def _build_result(det, cls_name, cls_ind, labels_csv):
    decodeScoreValue = encodeValue(
        label_dict=labels_csv, classIndex=int(cls_ind), value=float(det[-1]))
    ret = dict(index=cls_ind, score=decodeScoreValue)
    ret['class'] = cls_name
    x1, y1, x2, y2 = det[:4]
    ret['pts'] = [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]
    return ret


def parse_label_file(labelFile=None):
    label_dict = dict()
    with open(labelFile, 'r') as f:
        line_list = [i.strip() for i in f.readlines() if i]
        keyList = line_list[0].split(',')  # index,class,threshold
        for key in keyList[1:]:
            label_dict[key] = dict()
        for i_line in line_list[1:]:
            i_line_list = i_line.split(',')
            index_value = int(i_line_list[0])
            for colume_index, value in enumerate(i_line_list[1:], 1):
                label_dict[keyList[colume_index]][index_value] = value
    return label_dict


def encodeValue(label_dict=None, classIndex=None, value=None):
    """
    input :
            label_dict : dict ,labels.csv file parse result dict
            classIndex : int ,index of class in model
    return encoded value
    """
    classIndex = int(classIndex)
    value = float(value)
    minModelThreshold = float(label_dict['minMt'][classIndex])
    reviewModelThreshold = float(label_dict['revMt'][classIndex])
    minServingThreshold = float(label_dict['minSt'][classIndex])
    reviewServingThreshold = float(label_dict['revSt'][classIndex])
    resultValue = value
    if value <= minModelThreshold:
        # assert value > minModelThreshold
        resultValue = value
    elif value <= reviewModelThreshold:
        # review = True
        resultValue = (value - minModelThreshold) * (
            (reviewServingThreshold - minServingThreshold) /
            (reviewModelThreshold - minModelThreshold)) + minServingThreshold
    elif value > reviewModelThreshold:
        # review = False
        resultValue = (value - reviewModelThreshold) * (
            (1 - reviewServingThreshold) /
            (1 - reviewModelThreshold)) + reviewServingThreshold
    return resultValue


def decodeValue(label_dict=None, classIndex=None, value=None):
    """
        input :
            label_dict : labels.csv file parse result dict
            classIndex : index of class in model
        return decoded value
    """
    classIndex = int(classIndex)
    value = float(value)
    minModelThreshold = float(label_dict['minMt'][classIndex])
    reviewModelThreshold = float(label_dict['revMt'][classIndex])
    minServingThreshold = float(label_dict['minSt'][classIndex])
    reviewServingThreshold = float(label_dict['revSt'][classIndex])
    resultValue = value
    if value < minServingThreshold:
        resultValue = value
    elif value <= reviewServingThreshold:
        resultValue = (value - minServingThreshold) * (
            (reviewModelThreshold - minModelThreshold) /
            (reviewServingThreshold - minServingThreshold)) + minModelThreshold
    elif value > reviewServingThreshold:
        resultValue = (value - reviewServingThreshold) * (
            (1 - reviewModelThreshold) /
            (1 - reviewServingThreshold)) + reviewModelThreshold
    return resultValue


class InferenceServer(BaseInferenceServer):
    def __init__(self, app_name, cfg):
        super(InferenceServer, self).__init__(app_name)
        self.inference_req = InferenceReq()
        self.classes = aisdk.common.other.make_synset(
            cfg['model_files']['labels.csv'])
        label_file = cfg['model_files']['labels.csv']
        self.labels = parse_label_file(label_file)
        yaml_file = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'resnet.yaml')
        update_config(yaml_file)

    def net_inference(self, request):
        classes_dict = self.labels['class']
        threshold_dict = self.labels['minMt']  # minModelThreshold
        assert isinstance(request, pb.InferenceRequest)

        img = load_imagev2(request.data.body)
        assert img.ndim == 3  # TODO

        nms = py_nms_wrapper(config.TEST.NMS)

        if img.shape[0] > img.shape[1]:
            long_side, short_side = img.shape[0], img.shape[1]
        else:
            long_side, short_side = img.shape[1], img.shape[0]

        if short_side > 0 and float(long_side) / float(short_side) > 50.0:
            raise ErrorBase(
                400,
                'aspect ration is too large, long_size:short_side should not larger than 50.0'
            )

        batch = generate_batch(img)
        msg = pb.ForwardMsg()
        msg.network_input_buf = cPickle.dumps(
            batch, protocol=cPickle.HIGHEST_PROTOCOL)

        msg_out = self.inference_req.inference_msg(msg)
        scores = []
        boxes = []

        r = cPickle.loads(msg_out.network_output_buf)
        scores.append(r['scores'])
        boxes.append(r['boxes'])

        det_ret = []
        for cls_index in sorted(classes_dict.keys()):
            cls_ind = cls_index
            cls_name = classes_dict.get(cls_ind)
            cls_boxes = boxes[0][:, 4:8] if config.CLASS_AGNOSTIC else boxes[
                0][:, 4 * cls_ind:4 * 4 * (cls_ind + 1)]
            cls_scores = scores[0][:, cls_ind, np.newaxis]
            threshold = float(threshold_dict[cls_ind])
            keep = np.where(cls_scores > threshold)[0]
            dets = np.hstack((cls_boxes,
                              cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            det_ret.extend(
                _build_result(det, cls_name, cls_ind, self.labels)
                for det in dets[keep, :])
        return pb.InferenceResponse(
            code=200, result=json.dumps({
                'detections': det_ret
            }))


def serve():
    s = InferenceServer(const.app_name, const.cfg)
    s.serve()


if __name__ == '__main__':
    const.flavor.run_inference(serve)
