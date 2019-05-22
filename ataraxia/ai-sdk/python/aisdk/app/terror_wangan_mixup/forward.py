#coding=utf8
import six.moves.cPickle as cPickle  # pylint: disable=no-name-in-module,import-error
import time
import json
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format

from aisdk.framework.base_forward import BaseForwardServer
from aisdk.common.logger import log
import aisdk.proto as pb
from . import const


def change_deploy(deploy_file, input_data_batch_size):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(deploy_file).read(), net)
    data_layer_index = 0
    data_layer = net.layer[data_layer_index]
    data_layer.input_param.shape[0].dim[0] = input_data_batch_size
    new_deploy_file = deploy_file + '_tmp'
    with open(new_deploy_file, 'w') as f:
        f.write(str(net))
    return new_deploy_file


class ForwardServer(BaseForwardServer):
    def __init__(self, app_name, batch_size, lock, cfg):
        super(ForwardServer, self).__init__(app_name, batch_size)
        self.lock = lock

        caffe.set_mode_gpu()

        fine_deploy = str(cfg['model_files']["fine_deploy.prototxt"])
        fine_weight = str(cfg['model_files']["fine_weight.caffemodel"])

        det_deploy = str(cfg['model_files']['det_deploy.prototxt'])
        det_weight = str(cfg['model_files']['det_weight.caffemodel'])

        new_fine_deploy = change_deploy(fine_deploy, batch_size)
        new_det_deploy = change_deploy(det_deploy, batch_size)
        self.net_fine = caffe.Net(new_fine_deploy, fine_weight, caffe.TEST)
        self.net_det = caffe.Net(new_det_deploy, det_weight, caffe.TEST)

    def net_inference(self, msgs):
        assert isinstance(msgs, pb.ForwardMsgs)
        start = time.time()

        for index, msg in enumerate(msgs.msgs):
            r = cPickle.loads(msg.network_input_buf)
            img_cls = r['img_cls']
            assert img_cls.shape == (3, 225, 225)
            img_det = r['img_det']
            assert img_det.shape == (3, 512, 512)

            self.net_fine.blobs['data'].data[index] = img_cls
            self.net_det.blobs['data'].data[index] = img_det
        with self.lock:
            start_forward = time.time()
            output_fine = self.net_fine.forward()
            output_det = self.net_det.forward()
            end_forward = time.time()
        assert output_fine['prob'].shape[1:] == (48, 1, 1)
        # shape 第一维是 batch_size，第二维度是48类
        assert output_det['detection_out'].shape[1] == 1
        assert output_det['detection_out'].shape[3] == 7
        # shape 第一维是 batch_size，第三维度是检测到的物体数目，第四维度是类别
        buf = cPickle.dumps({
            'output_fine': output_fine,
            'output_det': output_det
        },
                            protocol=cPickle.HIGHEST_PROTOCOL)
        msgs_out = []
        for i in range(len(msgs.msgs)):
            msgs_out.append(
                pb.ForwardMsg(
                    network_output_buf=buf,
                    meta={
                        "data": json.dumps({
                            'image_index': i
                        }).encode('utf8')
                    }))
        log.info('{} use time {}, forward time {}, batch_size: {}/{}'.format(
            self.app_name,
            time.time() - start, end_forward - start_forward, len(msgs.msgs),
            self.batch_size))
        return pb.ForwardMsgs(msgs=msgs_out)


def serve(lock):
    srv = ForwardServer(const.app_name, const.cfg['batch_size'], lock,
                        const.cfg)
    srv.serve()


if __name__ == '__main__':
    const.flavor.run_forward(serve)
