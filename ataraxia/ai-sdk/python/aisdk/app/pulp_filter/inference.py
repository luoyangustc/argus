import json
import cv2
import numpy as np
import six.moves.cPickle as cPickle  # pylint: disable=no-name-in-module,import-error
import schema
import random

from aisdk.framework.base_inference import InferenceReq, BaseInferenceServer
from aisdk.common.image import load_imagev2
import aisdk.proto as pb
import aisdk.common.mxnet_base.net
from . import const

args_schema = schema.Schema({schema.Optional('limit'): int})

_map_pre_score = 0.89


class InferenceServer(BaseInferenceServer):
    def __init__(self, app_name, cfg):
        super(InferenceServer, self).__init__(app_name)
        net = aisdk.common.mxnet_base.net
        conf = net.NetConfig()
        conf.parse(cfg)
        label_file = conf.file_synset
        inference_req = InferenceReq()

        self.net = net
        self.labels = net.load_labels(label_file)
        self.image_width = conf.image_width
        self.image_height = conf.image_height
        self.mean_value = conf.value_mean
        self.std_value = conf.value_std
        self.inference_req = inference_req

    def net_inference(self, request):
        # pylint: disable=too-many-locals
        assert isinstance(request, pb.InferenceRequest)

        img = _load_image(request.data.body, self.image_width,
                          self.image_height, self.mean_value, self.std_value)
        msg = pb.ForwardMsg(
            network_input_buf=img.tobytes(), reqid=request.reqid)
        msg_out = self.inference_req.inference_msg(msg)
        output = cPickle.loads(msg_out.network_output_buf)
        return pb.InferenceResponse(
            code=200, result=json.dumps(_build_result(output, self.labels)))


def _load_image(body, width, height, mean_value, std_value):
    img = load_imagev2(body)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = cv2.resize(img, (width, height))
    img -= mean_value
    img /= std_value
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img


def _score_map(score):
    map_pre_score = _map_pre_score
    map_post_score = 0.6
    if score > map_pre_score:
        a = (1 - map_post_score) / (1 - map_pre_score)
        b = 1 - a
    else:
        a = map_post_score / map_pre_score
        b = 0
    score = (a * score + b)
    return score


def _build_result(output, labels):
    results = {"confidences": [], "checkpoint": None}

    label_dic = {0: 2, 2: 1, 1: 0}
    normal_score = float(output[0])
    if normal_score >= 0.960:
        a = random.uniform(0, 1)
        b = 1.0 - a
        mock = 1.0 - normal_score
        pulp_score = min(a, b) * mock
        sexy_score = max(a, b) * mock
        results["checkpoint"] = "endpoint"

        results["confidences"].append({
            'score': _score_map(float(output[0])),
            'class': labels[0][-1],
            'index': int(2)
        })
        results["confidences"].append({
            'score': _score_map(sexy_score),
            'class': labels[2][-1],
            'index': int(1)
        })
        results["confidences"].append({
            'score': _score_map(pulp_score),
            'class': labels[1][-1],
            'index': int(0)
        })

    else:
        results["checkpoint"] = "ava-pulp"
        j = np.argsort(output)[::-1]

        results["confidences"].append({
            'score':
            _score_map(float(output[j[0]])),
            'class':
            labels[j[0]][-1],
            'index':
            label_dic[int(labels[j[0]][0])]
        })
        results["confidences"].append({
            'score':
            _score_map(float(output[j[1]])),
            'class':
            labels[j[1]][-1],
            'index':
            label_dic[int(labels[j[1]][0])]
        })
        results["confidences"].append({
            'score': float(0),
            'class': 'sexy',
            'index': 1
        })

    return results


def serve():
    s = InferenceServer(const.app_name, const.cfg)
    s.serve()


if __name__ == '__main__':
    const.flavor.run_inference(serve)
