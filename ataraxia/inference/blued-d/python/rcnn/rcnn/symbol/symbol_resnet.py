import mxnet as mx
import proposal
import proposal_target
import roi_global_context
from rcnn.config import config

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
units = res_deps['101']
filter_list = [256, 512, 1024, 2048]

def get_shared_weights():
    shared_weights = {}
    _v = mx.symbol.Variable
    shared_weights['stage4_unit1_bn1_gamma'], shared_weights['stage4_unit1_bn1_beta']\
         = _v(name='stage4_unit1_bn1_gamma'), _v(name='stage4_unit1_bn1_beta')
    shared_weights['stage4_unit1_conv1_weight'] = _v(name='stage4_unit1_conv1_weight')
    shared_weights['stage4_unit1_bn2_gamma'], shared_weights['stage4_unit1_bn2_beta']\
         = _v(name='stage4_unit1_bn2_gamma'), _v(name='stage4_unit1_bn2_beta')
    shared_weights['stage4_unit1_conv2_weight'] = _v(name='stage4_unit1_conv2_weight')
    shared_weights['stage4_unit1_bn3_gamma'], shared_weights['stage4_unit1_bn3_beta']\
         = _v(name='stage4_unit1_bn3_gamma'), _v(name='stage4_unit1_bn3_beta')
    shared_weights['stage4_unit1_conv3_weight'] = _v(name='stage4_unit1_conv3_weight')
    shared_weights['stage4_unit1_sc_weight'] = _v(name='stage4_unit1_sc_weight')
    shared_weights['stage4_unit2_bn1_gamma'], shared_weights['stage4_unit2_bn1_beta']\
         = _v(name='stage4_unit2_bn1_gamma'), _v(name='stage4_unit2_bn1_beta')
    shared_weights['stage4_unit2_conv1_weight'] = _v(name='stage4_unit2_conv1_weight')
    shared_weights['stage4_unit2_bn2_gamma'], shared_weights['stage4_unit2_bn2_beta']\
         = _v(name='stage4_unit2_bn2_gamma'), _v(name='stage4_unit2_bn2_beta')
    shared_weights['stage4_unit2_conv2_weight'] = _v(name='stage4_unit2_conv2_weight')
    shared_weights['stage4_unit2_bn3_gamma'], shared_weights['stage4_unit2_bn3_beta']\
         = _v(name='stage4_unit2_bn3_gamma'), _v(name='stage4_unit2_bn3_beta')
    shared_weights['stage4_unit2_conv3_weight'] = _v(name='stage4_unit2_conv3_weight')
    shared_weights['stage4_unit3_bn1_gamma'], shared_weights['stage4_unit3_bn1_beta']\
         = _v(name='stage4_unit3_bn1_gamma'), _v(name='stage4_unit3_bn1_beta')
    shared_weights['stage4_unit3_conv1_weight'] = _v(name='stage4_unit3_conv1_weight')
    shared_weights['stage4_unit3_bn2_gamma'], shared_weights['stage4_unit3_bn2_beta']\
         = _v(name='stage4_unit3_bn2_gamma'), _v(name='stage4_unit3_bn2_beta')
    shared_weights['stage4_unit3_conv2_weight'] = _v(name='stage4_unit3_conv2_weight')
    shared_weights['stage4_unit3_bn3_gamma'], shared_weights['stage4_unit3_bn3_beta']\
         = _v(name='stage4_unit3_bn3_gamma'), _v(name='stage4_unit3_bn3_beta')
    shared_weights['stage4_unit3_conv3_weight'] = _v(name='stage4_unit3_conv3_weight')
    shared_weights['bn1_gamma'], shared_weights['bn1_beta'] = _v(name='bn1_gamma'), _v(name='bn1_beta')
    return shared_weights

def residual_unit(data, num_filter, stride, dim_match, name, sharing_weights=False, sharing_name=None, weights=None):
    if sharing_weights:
        if sharing_name is None:
            sharing_name = name
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1',\
                                gamma=weights[sharing_name + '_bn1_gamma'], beta=weights[sharing_name + '_bn1_beta'])
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1', weight=weights[sharing_name + '_conv1_weight'])
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2',\
                                gamma=weights[sharing_name + '_bn2_gamma'], beta=weights[sharing_name + '_bn2_beta'])
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2', weight=weights[sharing_name + '_conv2_weight'])
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3',\
                                gamma=weights[sharing_name + '_bn3_gamma'], beta=weights[sharing_name + '_bn3_beta'])
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3', weight=weights[sharing_name + '_conv3_weight'])
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc', weight=weights[sharing_name + '_sc_weight'])
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')

    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')

    return sum


def get_resnet_conv(data):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    return unit


def get_resnet_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS, use_global_context=False, use_roi_align=False):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)

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
        rois = mx.contrib.symbol.Proposal(
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
    if use_roi_align:
        roi_pool = mx.symbol.ROIAlign(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    else:
       roi_pool = mx.symbol.ROIPooling(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # res5
    unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
    bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    # res5
    if use_global_context:
        # weights to be shared
        shared_weights = get_shared_weights()

        # res5 for original rois
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1',\
                             sharing_weights=True, weights=shared_weights)
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i,\
                                 sharing_weights=True, weights=shared_weights)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1',\
                               gamma=shared_weights['bn1_gamma'], beta=shared_weights['bn1_beta'])
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

        #rois with global context
        rois_globalcontext = mx.symbol.Custom(rois=rois, im_info=im_info, global_context_scale=1.2,
                                              op_type='roi_global_context')
        if use_roi_align:
            roi_globalcontext_pool = mx.symbol.ROIAlign(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        else:
            roi_globalcontext_pool = mx.symbol.ROIPooling(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        # res5 for global context
        unit2 = residual_unit(data=roi_globalcontext_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage5_unit1',\
                              sharing_weights=True, sharing_name='stage4_unit1', weights=shared_weights)
        for i in range(2, units[3] + 1):
            unit2 = residual_unit(data=unit2, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage5_unit%s' % i,\
                                 sharing_weights=True, sharing_name='stage4_unit%s' % i, weights=shared_weights)
        bn2 = mx.sym.BatchNorm(data=unit2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn2',\
                               gamma=shared_weights['bn1_gamma'], beta=shared_weights['bn1_beta'])
        relu2 = mx.sym.Activation(data=bn2, act_type='relu', name='relu2')
        pool2 = mx.symbol.Pooling(data=relu2, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool2')
        # concat two res5 features
        pool_concat = mx.symbol.Concat(pool1, pool2, dim = 1)
        print("pool1:", pool1)
        print("pool2:", pool2)
        print("pool_concat:", pool_concat)

    else:
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    # classification
    if use_global_context:
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool_concat, num_hidden=num_classes)
    else:
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


def get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS, use_global_context=False, use_roi_align=False):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)

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
        rois = mx.contrib.symbol.Proposal(
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
    if use_roi_align:
        roi_pool = mx.symbol.ROIAlign(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    else:
        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # res5
    if use_global_context:
        # res5 for original rois
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

        #rois with global context
        rois_globalcontext = mx.symbol.Custom(rois=rois, im_info=im_info, global_context_scale=1.2,
                                              op_type='roi_global_context')
        if use_roi_align:
            roi_globalcontext_pool = mx.symbol.ROIAlign(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        else:
            roi_globalcontext_pool = mx.symbol.ROIPooling(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

        # res5 for global context
        unit2 = residual_unit(data=roi_globalcontext_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage5_unit1')
        for i in range(2, units[3] + 1):
            unit2 = residual_unit(data=unit2, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage5_unit%s' % i)
        bn2 = mx.sym.BatchNorm(data=unit2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn2')
        relu2 = mx.sym.Activation(data=bn2, act_type='relu', name='relu2')
        pool2 = mx.symbol.Pooling(data=relu2, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool2')
        # concat two res5 features
        pool_concat = mx.symbol.Concat(pool1, pool2, dim = 1)
        # print("pool1:", pool1)
        # print("pool2:", pool2)
        # print("pool_concat:", pool_concat)

    else:
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    # classification
    if use_global_context:
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool_concat, num_hidden=num_classes)
    else:
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)

    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group

def get_resnet_rpn(num_anchors=config.NUM_ANCHORS):
    """
    Region Proposal Network with resnet
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)

    # RPN
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
    cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1, name="cls_prob")
    # bounding box regression
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss])
    return group


def get_resnet_rpn_test(num_anchors=config.NUM_ANCHORS):
    """
    Region Proposal Network with restnet
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)

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
        group = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE)
    else:
        group = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.PROPOSAL_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.PROPOSAL_POST_NMS_TOP_N,
            threshold=config.TEST.PROPOSAL_NMS_THRESH, rpn_min_size=config.TEST.PROPOSAL_MIN_SIZE)
    # rois = group[0]
    # score = group[1]

    return group

def get_resnet_rcnn(num_classes=config.NUM_CLASSES, use_global_context=False, use_roi_align=False):
    """
    Fast R-CNN with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    rois = mx.symbol.Variable(name='rois')
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # reshape input
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
    label = mx.symbol.Reshape(data=label, shape=(-1, ), name='label_reshape')
    bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_classes), name='bbox_target_reshape')
    bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_classes), name='bbox_weight_reshape')

    conv_feat = get_resnet_conv(data)
    # Fast R-CNN
    if use_roi_align:
        roi_pool = mx.symbol.ROIAlign(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    else:
        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # res5
    if use_global_context:
        # weights to be shared
        shared_weights = get_shared_weights()

        # res5 for original rois
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False,
                             name='stage4_unit1',
                             sharing_weights=True, weights=shared_weights)
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                                 name='stage4_unit%s' % i,
                                 sharing_weights=True, weights=shared_weights)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1', \
                               gamma=shared_weights['bn1_gamma'], beta=shared_weights['bn1_beta'])
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

        # rois with global context
        rois_globalcontext = mx.symbol.Custom(rois=rois, im_info=im_info, global_context_scale=1.2,
                                              op_type='roi_global_context')
        if use_roi_align:
            roi_globalcontext_pool = mx.symbol.ROIAlign(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        else:
            roi_globalcontext_pool = mx.symbol.ROIPooling(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        # res5 for global context
        unit2 = residual_unit(data=roi_globalcontext_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False,
                              name='stage5_unit1',
                              sharing_weights=True, sharing_name='stage4_unit1', weights=shared_weights)
        for i in range(2, units[3] + 1):
            unit2 = residual_unit(data=unit2, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                                  name='stage5_unit%s' % i,
                                  sharing_weights=True, sharing_name='stage4_unit%s' % i, weights=shared_weights)
        bn2 = mx.sym.BatchNorm(data=unit2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn2',
                               gamma=shared_weights['bn1_gamma'], beta=shared_weights['bn1_beta'])
        relu2 = mx.sym.Activation(data=bn2, act_type='relu', name='relu2')
        pool2 = mx.symbol.Pooling(data=relu2, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool2')
        # concat two res5 features
        pool_concat = mx.symbol.Concat(pool1, pool2, dim=1)
        print("pool1:", pool1)
        print("pool2:", pool2)
        print("pool_concat:", pool_concat)

    else:
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False,
                             name='stage4_unit1')
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                                 name='stage4_unit%s' % i)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    # classification
    if use_global_context:
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool_concat, num_hidden=num_classes)
    else:
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)

    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group

def get_resnet_rcnn_test(num_classes=config.NUM_CLASSES, use_global_context=False, use_roi_align=False):
    """
    Fast R-CNN Network with VGG
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')

    # reshape rois
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

    conv_feat = get_resnet_conv(data)

    # Fast R-CNN
    if use_roi_align:
        roi_pool = mx.symbol.ROIAlign(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    else:
        roi_pool = mx.symbol.ROIPooling(
            name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14),
            spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    # res5
    if use_global_context:
        # res5 for original rois
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False,
                             name='stage4_unit1')
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                                 name='stage4_unit%s' % i)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

        # rois with global context
        rois_globalcontext = mx.symbol.Custom(rois=rois, im_info=im_info, global_context_scale=1.2,
                                              op_type='roi_global_context')
        if use_roi_align:
            roi_globalcontext_pool = mx.symbol.ROIAlign(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        else:
            roi_globalcontext_pool = mx.symbol.ROIPooling(
                name='roi_pool5_globcon', data=conv_feat, rois=rois_globalcontext, pooled_size=(14, 14),
                spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        # res5 for global context
        unit2 = residual_unit(data=roi_globalcontext_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False,
                              name='stage5_unit1')
        for i in range(2, units[3] + 1):
            unit2 = residual_unit(data=unit2, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                                  name='stage5_unit%s' % i)
        bn2 = mx.sym.BatchNorm(data=unit2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn2')
        relu2 = mx.sym.Activation(data=bn2, act_type='relu', name='relu2')
        pool2 = mx.symbol.Pooling(data=relu2, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool2')
        # concat two res5 features
        pool_concat = mx.symbol.Concat(pool1, pool2, dim=1)
        # print("pool1:", pool1)
        # print("pool2:", pool2)
        # print("pool_concat:", pool_concat)

    else:
        unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False,
                             name='stage4_unit1')
        for i in range(2, units[3] + 1):
            unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                                 name='stage4_unit%s' % i)
        bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    # classification
    if use_global_context:
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool_concat, num_hidden=num_classes)
    else:
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)

    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group
