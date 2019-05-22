"""
Contains the definition of the Inception Resnet V2 architecture.		
As described in http://arxiv.org/abs/1602.07261.		
Inception-v4, Inception-ResNet and the Impact of Residual Connections		
on Learning		
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi		
"""
import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config

eps = 0.001
use_global_stats = True

#mx.symbol.softmax()
def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True,name="",no_bias=True):
    """
    convFactory  contains the conv layer , batchnormal and relu 
    :param data: input data
    :param num_filter: number of conv layer
    :param kernel: kernal size 
    :param stride: stride number
    :param pad: 
    :param act_type: RELU or not 
    :param mirror_attr: 
    :param with_act: 
    :param name : convFactoryName 
    :return: filterd  data
    """
    if name == "":
        conv = mx.symbol.Convolution(
            data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
        # bn = mx.symbol.BatchNorm(data=conv)
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=eps, use_global_stats=use_global_stats)
        if with_act:
            act = mx.symbol.Activation(
                data=bn, act_type=act_type, attr=mirror_attr)
            return act
        else:
            return bn
    else:
        conv = mx.symbol.Convolution(
            data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,name=name,no_bias=no_bias)
        # bn = mx.symbol.BatchNorm(data=conv)
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,name=name+"_bn")
        if with_act:
            act = mx.symbol.Activation(
                data=bn, act_type=act_type, attr=mirror_attr,name=name + "_relu")
            return act
        else:
            return bn

def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={},name=""):# inception resnet
    tower_conv = ConvFactory(net, 32, (1, 1),name= name+"_1x1") #inception_resnet_v2_a1_1x1
    tower_conv1_0 = ConvFactory(net, 32, (1, 1),name= name+"_3x3_reduce") # #inception_resnet_v2_a1_3X3_reduce
    tower_conv1_1 = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1, 1),name= name+"_3x3")
    tower_conv2_0 = ConvFactory(net, 32, (1, 1),name=name+"_3x3_2_reduce") # inception_resnet_v2_a1_3x3_2_reduce
    tower_conv2_1 = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1, 1),name=name+"_3x3_2")#inception_resnet_v2_a1_3x3_2
    tower_conv2_2 = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1, 1),name=name+"_3x3_3")
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = mx.symbol.Convolution(
            data=tower_mixed, num_filter=input_num_channels, kernel=(1,1), stride=(1,1), pad=(0,0),name=name+"_up",no_bias=False)
    #tower_out = ConvFactory(
    #    tower_mixed, input_num_channels, (1, 1), with_act=False,no_bias=False,name=name+"_up")# "inception_resnet_v2_a1_up"

    net += scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={},name=""):
    tower_conv = ConvFactory(net, 192, (1, 1),name=name+"_1x1")
    tower_conv1_0 = ConvFactory(net, 128, (1, 1),name=name+"_1x7_reduce") #inception_resnet_v2_b1_1x7_reduce
    tower_conv1_1 = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(0, 3),name=name+"_1x7")
    tower_conv1_2 = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(3, 0),name=name+"_7x1")
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = mx.symbol.Convolution(
            data=tower_mixed, num_filter=input_num_channels, kernel=(1,1), stride=(1,1), pad=(0,0),name=name+"_up",no_bias=False)
 #   tower_out = ConvFactory(
 #       tower_mixed, input_num_channels, (1, 1), with_act=False,name=name+"_up",no_bias=False)#inception_resnet_v2_b1_up
    net += scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={},name=""):
    tower_conv = ConvFactory(net, 192, (1, 1),name=name+"_1x1") #inception_resnet_v2_c1_1x1
    tower_conv1_0 = ConvFactory(net, 192, (1, 1),name=name+"_1x3_reduce") #inception_resnet_v2_c1_1x3_reduce
    tower_conv1_1 = ConvFactory(tower_conv1_0, 224, (1, 3), pad=(0, 1),name=name+"_1x3")#inception_resnet_v2_c1_1x3
    tower_conv1_2 = ConvFactory(tower_conv1_1, 256, (3, 1), pad=(1, 0),name=name+"_3x1")
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = mx.symbol.Convolution(
            data=tower_mixed, num_filter=input_num_channels, kernel=(1,1), stride=(1,1), pad=(0,0),name=name+"_up",no_bias=False)
 #tower_out = ConvFactory(
 #       tower_mixed, input_num_channels, (1, 1), with_act=False,name=name+"_up",no_bias=False)#inception_resnet_v2_c1_up
    net += scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def repeat(inputs, repetitions, layer,name, *args, **kwargs):
    outputs = inputs
    for i in range(repetitions):
        outputs = layer(outputs,name=name.format(i+1), *args, **kwargs)
    return outputs



def get_inceptionresnet_conv(data):
    # inceptionresnet 1
    incption_1 = ConvFactory(data=data, num_filter=32,kernel=(3, 3),pad=(1,1), stride=(2, 2),name = "conv1_3x3_s2")  #[ ,32,149,149]
    # inceptionresnet 2
    conv2a_3_3 = ConvFactory(incption_1, 32, (3, 3), pad=(1, 1),name="conv2_3x3_s1") # reduce the size -1
    conv2b_3_3 = ConvFactory(conv2a_3_3, 64, (3, 3), pad=(1, 1),name="conv3_3x3_s1")
    incption_2 = mx.symbol.Pooling(
        data=conv2b_3_3, kernel=(3, 3), stride=(2, 2),pad=(1,1),pool_type='max') # [*,64,73,73]
    # inceptionresnet 3
    conv3a_1_1 = ConvFactory(incption_2, 80, (1, 1),name="conv4_3x3_reduce")
    conv3b_3_3 = ConvFactory(conv3a_1_1, 192, (3, 3),pad=(1,1),name="conv4_3x3")
    incption_3 = mx.symbol.Pooling(
        data=conv3b_3_3, kernel=(3, 3), stride=(2, 2),pad=(1,1), pool_type='max')  # [*,192,35,35]
    # inceptionresnet 4
    tower_conv = ConvFactory(incption_3, 96, (1, 1),name="conv5_1x1")
    tower_conv1_0 = ConvFactory(incption_3, 48, (1, 1),name= "conv5_5x5_reduce")
    tower_conv1_1 = ConvFactory(tower_conv1_0, 64, (5, 5), pad=(2, 2),name="conv5_5x5")

    tower_conv2_0 = ConvFactory(incption_3, 64, (1, 1),name="conv5_3x3_reduce")
    tower_conv2_1 = ConvFactory(tower_conv2_0, 96, (3, 3), pad=(1, 1),name="conv5_3x3")
    tower_conv2_2 = ConvFactory(tower_conv2_1, 96, (3, 3), pad=(1, 1),name="conv5_3x3_2")

    tower_pool3_0 = mx.symbol.Pooling(data=incption_3, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    tower_conv3_1 = ConvFactory(tower_pool3_0, 64, (1, 1),name="conv5_1x1_ave")
    stem_inception_4 = mx.symbol.Concat(
        *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])  # [*,320,35,35]
        ## resnet begin
    res_out_4 = repeat(stem_inception_4, 10, block35, scale=0.17, input_num_channels=320,name="inception_resnet_v2_a{0}")
        #upscale and pooling
    tower_conv = ConvFactory(res_out_4, 384, (3, 3), stride=(2, 2),pad=(1,1),name="reduction_a_3x3")
    tower_conv1_0 = ConvFactory(res_out_4, 256, (1, 1),name="reduction_a_3x3_2_reduce")
    tower_conv1_1 = ConvFactory(tower_conv1_0, 256, (3, 3), pad=(1, 1),name="reduction_a_3x3_2")
    tower_conv1_2 = ConvFactory(tower_conv1_1, 384, (3, 3), pad=(1, 1),stride=(2, 2),name="reduction_a_3x3_3")
    tower_pool = mx.symbol.Pooling(res_out_4, kernel=(
        3, 3), stride=(2, 2),pad=(1,1),pool_type='max')
    incption_4 = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool]) # [*,1088,17,17]
    #inception_4 = incption_4
    inception_4 = repeat(incption_4, 20, block17, scale=0.1, input_num_channels=1088,name="inception_resnet_v2_b{0}")
    return inception_4


def get_inceptionresnet_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feat = get_inceptionresnet_conv(data)

    #print(conv_feat)
    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(17, 17), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    #inception 5
    net = roi_pool
    tower_conv = ConvFactory(net, 256, (1, 1),name="reduction_b_3x3_reduce")
    tower_conv0_1 = ConvFactory(tower_conv, 384, (3, 3), stride=(2, 2),name="reduction_b_3x3")
    tower_conv1 = ConvFactory(net, 256, (1, 1),name="reduction_b_3x3_2_reduce")
    tower_conv1_1 = ConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2),name="reduction_b_3x3_2")
    tower_conv2 = ConvFactory(net, 256, (1, 1),name="reduction_b_3x3_3_reduce")
    tower_conv2_1 = ConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1),name="reduction_b_3x3_3")
    tower_conv2_2 = ConvFactory(tower_conv2_1, 320, (3, 3),  stride=(2, 2),name="reduction_b_3x3_4")
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(
        *[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])
    # inception 6
    net = repeat(net, 9, block8, scale=0.2, input_num_channels=2080,name="inception_resnet_v2_c{0}")
    net = block8(net, with_act=False, input_num_channels=2080,name="inception_resnet_v2_c10")

    net = ConvFactory(net, 1536, (1, 1),name="conv6_1x1")
    pool1 = mx.symbol.Pooling(net, kernel=(
        8, 8), global_pool=True, pool_type='avg')
    pool1 = mx.symbol.Flatten(pool1)
    pool1 = mx.symbol.Dropout(data=pool1, p=0.2)
   # pool1 = mx.symbol.FullyConnected(data=pool1, num_hidden=num_classes)

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group


def get_inceptionresnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_inceptionresnet_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(17, 17), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    net = roi_pool
    # inception 5
    # net = repeat(roi_pool, 20, block17, scale=0.1, input_num_channels=1088, name="inception_resnet_v2_b{0}")
    tower_conv = ConvFactory(net, 256, (1, 1), name="reduction_b_3x3_reduce")
    tower_conv0_1 = ConvFactory(tower_conv, 384, (3, 3), stride=(2, 2), name="reduction_b_3x3")
    tower_conv1 = ConvFactory(net, 256, (1, 1), name="reduction_b_3x3_2_reduce")
    tower_conv1_1 = ConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2), name="reduction_b_3x3_2")
    tower_conv2 = ConvFactory(net, 256, (1, 1), name="reduction_b_3x3_3_reduce")
    tower_conv2_1 = ConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1), name="reduction_b_3x3_3")
    tower_conv2_2 = ConvFactory(tower_conv2_1, 320, (3, 3), stride=(2, 2), name="reduction_b_3x3_4")
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(
        *[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])
    # inception 6
#    net = repeat(net, 9, block8, scale=0.2, input_num_channels=2080, name="inception_resnet_v2_c{0}")
#    net = block8(net, with_act=False, input_num_channels=2080, name="inception_resnet_v2_c10")

    net = ConvFactory(net, 1536, (1, 1), name="conv6_1x1")
    pool1 = mx.symbol.Pooling(net, kernel=(
        8, 8), global_pool=True, pool_type='avg')
    pool1 = mx.symbol.Flatten(pool1)
    #pool1 = mx.symbol.Dropout(data=pool1, p=0.5)
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group


