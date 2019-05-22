import mxnet as mx
from rcnn.config import config

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {
    '50': (3, 4, 6, 3),
    '101': (3, 4, 23, 3),
    '152': (3, 8, 36, 3),
    '200': (3, 24, 36, 3)
}
units = res_deps['101']
filter_list = [256, 512, 1024, 2048]


def residual_unit(data, num_filter, stride, dim_match, name, wbs={}):
    bn1 = mx.sym.BatchNorm( data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    if 'conv1' in wbs:
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25),
            kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace,
            weight=wbs['conv1'].weight, name=name + '_conv1')
    else:
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25),
            kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv1')

    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps,
        use_global_stats=use_global_stats, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    if name+'_conv2' in wbs:
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25),
            kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True,
            workspace=workspace, weight=wbs[name+'_conv2']['weight'], name=name + '_conv2')
    else:
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25),
            kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True,
            workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps,
        use_global_stats=use_global_stats, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    if name+'_conv3' in wbs:
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1),
            stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace,
            weight=wbs[name+'_conv3']['weight'], name=name + '_conv3')
    else:
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1),
            stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv3')

    if dim_match:
        shortcut = data
    else:
        if name+'_sc' in wbs:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1),
                stride=stride, no_bias=True, workspace=workspace, weight=wbs[name+'_sc']['weight'],
                name=name + '_sc')
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter,
                kernel=(1, 1), stride=stride, no_bias=True, workspace=workspace,
                name=name + '_sc')

    sum = mx.sym.ElementWiseSum(* [conv3, shortcut], name=name + '_plus')
    return sum


def _build_features(features):
    '''Parameter: list of feature from resnet layers'''

    c2, c3, c4, c5 = features
    p5 = mx.sym.Convolution(
        data=c5, num_filter=256, kernel=(1, 1), name='p5_lateral')

    topdown = mx.sym.UpSampling(
        p5, scale=2, name='p4_topdown', sample_type='nearest')
    lateral = mx.sym.Convolution(
        data=c4, num_filter=256, kernel=(1, 1), name='p4_lateral')
    p4 = mx.sym.ElementWiseSum(* [topdown, lateral], name= 'p4_plus')
    p4 = mx.sym.Convolution(
        data=p4, num_filter=256, kernel=(3, 3),pad=(1,1), name='p4_conv')

    topdown = mx.sym.UpSampling(
        p4, scale=2, name='p3_topdown', sample_type='nearest')
    lateral = mx.sym.Convolution(
        data=c3, num_filter=256, kernel=(1, 1), name='p3_lateral')
    p3 = mx.sym.ElementWiseSum(* [topdown, lateral], name= 'p3_plus')
    p3 = mx.sym.Convolution(data=p3, num_filter=256, kernel=(3, 3), pad=(1, 1), name='p3_conv')

    topdown = mx.sym.UpSampling(p3, scale=2, name='p2_topdown', sample_type='nearest')
    lateral = mx.sym.Convolution(data=c2, num_filter=256, kernel=(1, 1), name='p2_lateral')
    p2  = mx.sym.ElementWiseSum(* [topdown, lateral], name= 'p2_plus')
    p2 = mx.sym.Convolution(data=p2, num_filter=256, kernel=(3, 3), pad=(1, 1), name='p2_conv')

    return p2, p3, p4, p5


def get_resnet_fpn_conv(data):
    # res1
    data_bn = mx.sym.BatchNorm(data=data,fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling( data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    features = []
    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)
    features.append(unit)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)
    features.append(unit)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    features.append(unit)

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
    features.append(unit)

    return _build_features(features)
#    return features


def get_resnet_fpn_train(num_classes=config.NUM_CLASSES,
                         num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
#    rpn_label = mx.symbol.Variable(name='label')
#    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
#    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    rpn_label = [mx.symbol.Variable(name='labelp2'),mx.symbol.Variable(name='labelp3'),mx.symbol.Variable(name='labelp4'),mx.symbol.Variable(name='labelp5')]
    rpn_bbox_target = [mx.symbol.Variable(name='bbox_targetp2'),mx.symbol.Variable(name='bbox_targetp3'),mx.symbol.Variable(name='bbox_targetp4'),mx.symbol.Variable(name='bbox_targetp5')]
    rpn_bbox_weight = [mx.symbol.Variable(name='bbox_weightp2'),mx.symbol.Variable(name='bbox_weightp3'),mx.symbol.Variable(name='bbox_weightp4'),mx.symbol.Variable(name='bbox_weightp5')]
 

    scales = ( 4, 8, 16, 32)
    rpn_cls_probes, rpn_bbox_losses, cls_probes, bbox_losses, labels = [], [], [], [], []

    _v = mx.symbol.Variable
    rpn_conv_weight, rpn_conv_bias = _v(name='rpn_conv_3x3_weight'), _v(
        name='rpn_conv_3x3_bias')
    rpn_cls_score_weight, rpn_cls_score_bias = _v(
        name='rpn_cls_score_weight'), _v(name='rpn_cls_score_bias')
    rpn_bbox_pred_weight, rpn_bbox_pred_bias = _v(
        name='rpn_bbox_pred_weight'), _v(name='rpn_bbox_pred_bias')
    rpn_cls_score_weight, rpn_cls_score_bias = _v(
        name='rpn_cls_score_weight'), _v(name='rpn_cls_score_bias')
    bbox_pred_weight, bbox_pred_bias = _v(name='bbox_pred_weight'), _v(
        name='bbox_pred_bias')
    cls_score_weight, cls_score_bias = _v(name='cls_score_weight'), _v(
        name='cls_score_bias')
   
    fc1_weight, fc1_bias = _v(name='fc1_weight'), _v(
        name='fc1_bias')


#    p = (get_resnet_fpn_conv(data))
#    return p[2]
    # shared convolutional layers


    for i, conv_feat in enumerate(get_resnet_fpn_conv(data)):
        # RPN layers
        rpn_conv = mx.symbol.Convolution(data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, weight=rpn_conv_weight, bias=rpn_conv_bias, name="rpn_conv_3x3")
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu_%d" % i)
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, weight=rpn_cls_score_weight, bias=rpn_cls_score_bias, name="rpn_cls_score_%d" % i)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, weight=rpn_bbox_pred_weight, bias=rpn_bbox_pred_bias, name="rpn_bbox_pred_%d" % i)

        # prepare rpn data
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_%d" % i)

        # classification
        rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label[i], multi_output=True, normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob_%d" % i)
        # bounding box regression
        rpn_bbox_loss_ = rpn_bbox_weight[i] * mx.symbol.smooth_l1(name='rpn_bbox_loss_%d' % i, scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target[i]))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss_%d' % i, data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

        # ROI proposal
        rpn_cls_act = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act_%d" % i)
        rpn_cls_act_reshape = mx.symbol.Reshape(data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_%d' % i)

        if config.TRAIN.CXX_PROPOSAL:
          #  print("processing")
            rois = mx.contrib.symbol.Proposal(cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_%d' % i,
                feature_stride=scales[i], scales=(16, ), ratios=tuple(config.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
        else:
            #print("processing")
            rois = mx.symbol.Custom(cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info,
                name='rois_%d' % i, op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE[i], scales=(16, ),
                ratios=tuple(config.ANCHOR_RATIOS), rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N,
                rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N, threshold=config.TRAIN.RPN_NMS_THRESH,
                rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

        # ROI proposal target
        gt_boxes_reshape = mx.symbol.Reshape(
            data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape_%d' % i)
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target', num_classes=num_classes,
            batch_images=config.TRAIN.BATCH_IMAGES, batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)

        #rois, label, bbox_target, bbox_weight = group[:4]

        rois=group[0]
        label = group[1]
        bbox_target = group[2]
        bbox_weight = group[3]


        # Fast R-CNN
        roi_pool = mx.symbol.ROIPooling(name='roi_pool5_%d' % i, data=conv_feat, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / scales[i])
        # res5
        #unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1', wbs=residua_unit5_weight)
        #for i in range(2, units[3] + 1):
        #    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i, wbs=residua_unit5_weight)

        #bn1 = mx.sym.BatchNorm(data=roi_pool, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
        #relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        #pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

        fc1 = mx.symbol.FullyConnected(name='fc1', data=roi_pool, num_hidden=1024, weight=fc1_weight, bias=fc1_bias)
        # classification
        cls_score = mx.symbol.FullyConnected(name='cls_score_%d' % i, data=fc1, num_hidden=num_classes, weight=cls_score_weight, bias=cls_score_bias)
        cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob_%d' % i, data=cls_score, label=label, normalization='batch')
        # bounding box regression
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred_%d' % i, data=fc1, num_hidden=num_classes * 4, weight=bbox_pred_weight, bias=bbox_pred_bias)
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_%d' % i, scalar=1.0, data=(bbox_pred - bbox_target))
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss_%d' % i, data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

        # reshape output
        label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape_%d' % i)
        cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape_%d' % i)
        bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape_%d' % i)

        rpn_cls_probes.append(rpn_cls_prob)
        rpn_bbox_losses.append(rpn_bbox_loss)
        cls_probes.append(cls_prob)
        bbox_losses.append(bbox_loss)
        labels.append(label)


#    return cls_probes[1]
    _s = mx.sym.ElementWiseSum
  #  p2  = mx.sym.ElementWiseSum(* [topdown, lateral], name= 'p2_plus')
    #group = mx.symbol.Group([_c(*rpn_cls_probes, dim=0), _s(*rpn_bbox_losses), _c(*cls_probes, dim=0), _s(*bbox_losses), mx.symbol.BlockGrad(_c(*labels, dim=0))])
    test = [rpn_cls_prob for rpn_cls_prob in rpn_cls_probes]
    #test.append(_s(*rpn_bbox_losses))
    test.append(rpn_bbox_losses[0])
    test.append(rpn_bbox_losses[1])
    test.append(rpn_bbox_losses[2])
    test.append(rpn_bbox_losses[3])
    for cls_probe in cls_probes:
        test.append(cls_probe) 
    test.append(_s(*bbox_losses))
    for label in labels:
        test.append(label)
    #print(test)
    group = mx.symbol.Group(test)
    
    return group


def get_resnet_fpn_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_fpn_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=conv_feat,
        kernel=(3, 3),
        pad=(1, 1),
        num_filter=512,
        name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(
        data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu,
        kernel=(1, 1),
        pad=(0, 0),
        num_filter=2 * num_anchors,
        name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu,
        kernel=(1, 1),
        pad=(0, 0),
        num_filter=4 * num_anchors,
        name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob,
        shape=(0, 2 * num_anchors, -1, 0),
        name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape,
            bbox_pred=rpn_bbox_pred,
            im_info=im_info,
            name='rois',
            feature_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N,
            rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH,
            rpn_min_size=config.TEST.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape,
            bbox_pred=rpn_bbox_pred,
            im_info=im_info,
            name='rois',
            op_type='proposal',
            feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N,
            rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH,
            rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool5',
        data=conv_feat,
        rois=rois,
        pooled_size=(14, 14),
        spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    # res5
    unit = residual_unit(
        data=roi_pool,
        num_filter=filter_list[3],
        stride=(2, 2),
        dim_match=False,
        name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(
            data=unit,
            num_filter=filter_list[3],
            stride=(1, 1),
            dim_match=True,
            name='stage4_unit%s' % i)
    bn1 = mx.sym.BatchNorm(
        data=unit,
        fix_gamma=False,
        eps=eps,
        use_global_stats=use_global_stats,
        name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(
        data=relu1,
        global_pool=True,
        kernel=(7, 7),
        pool_type='avg',
        name='pool1')

    # classification
    cls_score = mx.symbol.FullyConnected(
        name='cls_score', data=pool1, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(
        name='bbox_pred', data=pool1, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(
        data=cls_prob,
        shape=(config.TEST.BATCH_IMAGES, -1, num_classes),
        name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(
        data=bbox_pred,
        shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes),
        name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group
