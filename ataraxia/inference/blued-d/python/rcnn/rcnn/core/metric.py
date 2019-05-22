import mxnet as mx
import numpy as np

from rcnn.config import config


def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names():
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if config.TRAIN.END2END:
        pred.append('rcnn_label')
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        #print(label)
        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)



class MutilTaskAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MutilTaskAccMetric, self).__init__('MTAcc')
#        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()
        self.pred.append('mutiltask_cls')
        self.label.append('gtlabel')

    def update(self, labels, preds):
        pred = preds[self.pred.index('mutiltask_cls')]
        label = labels[self.label.index('gtlabel')]


        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')
        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')
        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst
        keep_inds = np.where(label != 0)[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


def get_rpn_names_fpn():
    pred = ['rpn_cls_prob_0', 'rpn_cls_prob_1','rpn_cls_prob_2','rpn_cls_prob_3',
            'rpn_bbox_loss_0','rpn_bbox_loss_1','rpn_bbox_loss_2','rpn_bbox_loss_3']
    label = ['rpn_label_0','rpn_bbox_target_0','rpn_bbox_weight','rpn_label_1','rpn_bbox_target_1','rpn_bbox_weight','rpn_label_2',
             'rpn_bbox_target_2','rpn_bbox_weight','rpn_label_3','rpn_bbox_target_3',
             'rpn_bbox_weight']
#    label = ['rpn_label_0','rpn_label_1','rpn_label_2','rpn_label_3',
#             'rpn_bbox_target_0','rpn_bbox_target_1','rpn_bbox_target_2','rpn_bbox_target_3',
#             'rpn_bbox_weight']
    return pred, label


def get_rcnn_names_fpn():
    pred = ['rcnn_cls_prob_0', 'rcnn_cls_prob_1','rcnn_cls_prob_2','rcnn_cls_prob_3',
            'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if config.TRAIN.END2END:
        pred.append('rcnn_label_0')
        pred.append('rcnn_label_1')
        pred.append('rcnn_label_2')
        pred.append('rcnn_label_3')
        rpn_pred, rpn_label = get_rpn_names_fpn()
        #nprint(rpn_label[0:1])
        pred = rpn_pred + pred
        #label = rpn_label[0:2] + ['test'] + rpn_label[2:4] + ['test'] + rpn_label[4:6] + ['test'] + rpn_label[6:8] + ['test']
        label = rpn_label

    return pred, label



class FPNRPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRPNAccMetric, self).__init__('FPNRPNAcc')
        if config.TRAIN.END2END:
            self.pred, self.label = get_rcnn_names_fpn()
        else:
            self.pred, self.label = get_rpn_names_fpn()


    def update(self, labels, preds):
        #print(labels)
        pred_0 = preds[self.pred.index('rpn_cls_prob_0')]
        label_0 = labels[self.label.index('rpn_label_0')]
        pred_1 = preds[self.pred.index('rpn_cls_prob_1')]
        label_1 = labels[self.label.index('rpn_label_1')]
        pred_2 = preds[self.pred.index('rpn_cls_prob_2')]
        label_2 = labels[self.label.index('rpn_label_2')]
        pred_3 = preds[self.pred.index('rpn_cls_prob_3')]
        label_3 = labels[self.label.index('rpn_label_3')]

        fpn_preds = [pred_0,pred_1,pred_2,pred_3]
        fpn_labels = [label_0, label_1, label_2, label_3]
        # pred (b, c, p) or (b, c, h, w)
        for i in range(len(fpn_preds)):
            pred_label = mx.ndarray.argmax_channel(fpn_preds[i]).asnumpy().astype('int32')
            pred_label = pred_label.reshape((pred_label.shape[0], -1))
            # label (b, p)
            label = fpn_labels[i].asnumpy().astype('int32')

            #print(label.shape)
            #print(pred_label.shape)
        # filter with keep_inds
            keep_inds = np.where(label != -1)
            pred_label = pred_label[keep_inds]
            label = label[keep_inds]

            self.sum_metric += np.sum(pred_label.flat == label.flat)
            self.num_inst += len(pred_label.flat)


class FPNRCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRCNNAccMetric, self).__init__('FPNRCNNAcc')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names_fpn()

    def update(self, labels, preds):
        pred_0 = preds[self.pred.index('rcnn_cls_prob_0')]
        pred_1 = preds[self.pred.index('rcnn_cls_prob_1')]
        pred_2 = preds[self.pred.index('rcnn_cls_prob_2')]
        pred_3 = preds[self.pred.index('rcnn_cls_prob_3')]

        if self.e2e:
            label_0 = preds[self.pred.index('rcnn_label_0')]
            label_1 = preds[self.pred.index('rcnn_label_1')]
            label_2 = preds[self.pred.index('rcnn_label_2')]
            label_3 = preds[self.pred.index('rcnn_label_3')]
        else:
            label = labels[self.label.index('rcnn_label')]

        fpn_preds = [pred_0,pred_1,pred_2,pred_3]
        fpn_labels = [label_0, label_1, label_2, label_3]

        for i in range(len(fpn_preds)):
            last_dim = fpn_preds[i].shape[-1]
            pred_label = fpn_preds[i].asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
            label = fpn_labels[i].asnumpy().reshape(-1,).astype('int32')

            #print(label)
            self.sum_metric += np.sum(pred_label.flat == label.flat)
            self.num_inst += len(pred_label.flat)

class FPNRPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRPNLogLossMetric, self).__init__('FPNRPNLogLoss')
        if config.TRAIN.END2END:
            self.pred, self.label = get_rcnn_names_fpn()
        else:
            self.pred, self.label = get_rpn_names_fpn()


    def update(self, labels, preds):

        pred_0 = preds[self.pred.index('rpn_cls_prob_0')]
        label_0 = labels[self.label.index('rpn_label_0')]
        pred_1 = preds[self.pred.index('rpn_cls_prob_1')]
        label_1 = labels[self.label.index('rpn_label_1')]
        pred_2 = preds[self.pred.index('rpn_cls_prob_2')]
        label_2 = labels[self.label.index('rpn_label_2')]
        pred_3 = preds[self.pred.index('rpn_cls_prob_3')]
        label_3 = labels[self.label.index('rpn_label_3')]

        fpn_preds = [pred_0,pred_1,pred_2,pred_3]
        fpn_labels = [label_0, label_1, label_2, label_3]

        for i in range(len(fpn_preds)):
            # label (b, p)
            label = fpn_labels[i].asnumpy().astype('int32').reshape((-1))
            # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
            pred = fpn_preds[i].asnumpy().reshape((fpn_preds[i].shape[0],
                                                   fpn_preds[i].shape[1], -1)).transpose((0, 2, 1))
            pred = pred.reshape((label.shape[0], -1))

            # filter with keep_inds
            keep_inds = np.where(label != -1)[0]
            label = label[keep_inds]
            cls = pred[keep_inds, label]

            cls += 1e-14
            cls_loss = -1 * np.log(cls)
            cls_loss = np.sum(cls_loss)
            self.sum_metric += cls_loss
            self.num_inst += label.shape[0]


class FPNRCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRCNNLogLossMetric, self).__init__('FPNRCNNLogLoss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names_fpn()

    def update(self, labels, preds):
        #print(self.pred)
        #print(self.pred.index('rcnn_cls_prob_0'))
        #print(preds)
        #print("preds")
        #print("labels")
        pred_0 = preds[self.pred.index('rcnn_cls_prob_0')]
        pred_1 = preds[self.pred.index('rcnn_cls_prob_1')]
        pred_2 = preds[self.pred.index('rcnn_cls_prob_2')]
        pred_3 = preds[self.pred.index('rcnn_cls_prob_3')]

        if self.e2e:
            label_0 = preds[self.pred.index('rcnn_label_0')]
            label_1 = preds[self.pred.index('rcnn_label_1')]
            label_2 = preds[self.pred.index('rcnn_label_2')]
            label_3 = preds[self.pred.index('rcnn_label_3')]
        else:
            label = labels[self.label.index('rcnn_label')]

        fpn_preds = [pred_0, pred_1, pred_2, pred_3]
        fpn_labels = [label_0, label_1, label_2, label_3]
        for i in range(len(fpn_preds)):
            last_dim = fpn_preds[i].shape[-1]
            pred = fpn_preds[i].asnumpy().reshape(-1, last_dim)
            label = fpn_labels[i].asnumpy().reshape(-1,).astype('int32')
            cls = pred[np.arange(label.shape[0]), label]

            cls += 1e-14
            cls_loss = -1 * np.log(cls)
            cls_loss = np.sum(cls_loss)
            self.sum_metric += cls_loss
            self.num_inst += label.shape[0]


class FPNRPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRPNL1LossMetric, self).__init__('FPNRPNL1Loss')
        if config.TRAIN.END2END:
            self.pred, self.label = get_rcnn_names_fpn()
        else:
            self.pred, self.label = get_rpn_names_fpn()


    def update(self, labels, preds):

        bbox_loss_0 = preds[self.pred.index('rpn_bbox_loss_0')].asnumpy()
        bbox_loss_1 = preds[self.pred.index('rpn_bbox_loss_1')].asnumpy()
        bbox_loss_2 = preds[self.pred.index('rpn_bbox_loss_2')].asnumpy()
        bbox_loss_3 = preds[self.pred.index('rpn_bbox_loss_3')].asnumpy()

        fpn_bbox_losses = [bbox_loss_0,bbox_loss_1,bbox_loss_2,bbox_loss_3]

        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        for i in range(len(fpn_bbox_losses)):
            # calculate num_inst (average on those fg anchors)
            num_inst = np.sum(bbox_weight > 0) / 4

            self.sum_metric += np.sum(fpn_bbox_losses[i])
            self.num_inst += num_inst


class FPNRCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRCNNL1LossMetric, self).__init__('FPNRCNNL1Loss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names_fpn()

    def update(self, labels, preds):
        bbox_loss_0 = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        bbox_loss_1 = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        bbox_loss_2 = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        bbox_loss_3 = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        fpn_bbox_losses = [bbox_loss_0,bbox_loss_1,bbox_loss_2,bbox_loss_3]

        if self.e2e:
            label_0 = preds[self.pred.index('rcnn_label_0')].asnumpy()
            label_1 = preds[self.pred.index('rcnn_label_1')].asnumpy()
            label_2 = preds[self.pred.index('rcnn_label_2')].asnumpy()
            label_3 = preds[self.pred.index('rcnn_label_3')].asnumpy()
            fpn_labels = [label_0, label_1, label_2, label_3]
        else:
            label = labels[self.label.index('rcnn_label')].asnumpy()

        for i in range(len(fpn_bbox_losses)):
            # calculate num_inst
            keep_inds = np.where(fpn_labels[i] != 0)[0]
            num_inst = len(keep_inds)

            self.sum_metric += np.sum(fpn_bbox_losses[i])
            self.num_inst += num_inst

