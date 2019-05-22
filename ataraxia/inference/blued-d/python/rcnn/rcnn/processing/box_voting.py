import numpy as np

DEBUG = True

def py_box_voting_wrapper(IOU_thresh, score_thresh, with_nms):
    if with_nms:
        def _box_voting(nms_dets, dets):
            return box_voting_nms(nms_dets, dets, IOU_thresh, score_thresh)
    else:
        def _box_voting(dets):
            return box_voting(dets, IOU_thresh, score_thresh)
    return _box_voting

def box_voting_nms(nms_dets, dets, IOU_thresh, score_thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum 
    and voting the final box coordinates by fusing those boxes
    :param num_dets: dets after nms
    :param dets: original detection results, dets before nms. [[x1, y1, x2, y2 score]]
    :param IOU_thresh: retain overlap > IOU_thresh for fusion
    :param score_thresh: retain score > score_thresh for fusion
    :return: detection coordinates to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    if DEBUG:
        print("dets ordered:", dets[order])

    keep_fusion_boxes = []
    for idx, nms_det in enumerate(nms_dets):
        area_nms_det = (nms_det[2] - nms_det[0] + 1) * (nms_det[3] - nms_det[1] + 1)
        xx1 = np.maximum(nms_det[0], x1[order])
        yy1 = np.maximum(nms_det[1], y1[order])
        xx2 = np.minimum(nms_det[2], x2[order])
        yy2 = np.minimum(nms_det[3], y2[order])

        # compute overlap
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (area_nms_det + areas[order] - inter)

        # retain boxes with large overlap and high confidence for fusion
        IOU_inds_keep = np.where(ovr > IOU_thresh)[0]
        scores_inds_keep = np.where(scores[order] > score_thresh)[0]
        if DEBUG:
            print("IOU_inds_keep:", IOU_inds_keep)
            print("scores_inds_keep:", scores_inds_keep)

        inds_fusion = np.intersect1d(IOU_inds_keep, scores_inds_keep)
        if inds_fusion.size == 0: # if no box retained, keep the original one
            keep_fusion_boxes.append(nms_det)
            if DEBUG:
                print("inds_fusion:", inds_fusion)
                print("keep nms_det")
            continue

        if DEBUG:
            if inds_fusion.size>1:
                print("boxes for fusion:", inds_fusion)
                print(dets[order[inds_fusion]])

        x1_fusion = x1[order[inds_fusion]]
        y1_fusion = y1[order[inds_fusion]]
        x2_fusion = x2[order[inds_fusion]]
        y2_fusion = y2[order[inds_fusion]]
        scores_fusion = scores[order[inds_fusion]]
        fusion_box = np.zeros((5))
        fusion_box[0] = np.sum(x1_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[1] = np.sum(y1_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[2] = np.sum(x2_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[3] = np.sum(y2_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[4] = scores_fusion[0]
        if DEBUG:
            print("fusion_box:", fusion_box)

        keep_fusion_boxes.append(fusion_box)

        # boxes with small overlap are kept for another loop
        inds_next = np.where(ovr <= IOU_thresh)[0]
        order = order[inds_next]

    keep_fusion_boxes = np.array(keep_fusion_boxes)

    return keep_fusion_boxes


def box_voting(dets, IOU_thresh, score_thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum 
    and voting the final box coordinates by fusing those boxes
    :param num_dets: dets after nms
    :param dets: original detection results, dets before nms. [[x1, y1, x2, y2 score]]
    :param IOU_thresh: retain overlap > IOU_thresh for fusion
    :param score_thresh: retain score > score_thresh for fusion
    :return: detection coordinates to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    if DEBUG:
        print("dets ordered:", dets)

    keep_fusion_boxes = []
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        # compute overlap
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i]+ areas[order] - inter)

        # retain boxes with large overlap and high confidence for fusion
        IOU_inds_keep = np.where(ovr > IOU_thresh)[0]
        scores_inds_keep = np.where(scores[order] > score_thresh)[0]
        if DEBUG:
            print("IOU_inds_keep:", IOU_inds_keep)
            print("scores_inds_keep:", scores_inds_keep)

        if IOU_inds_keep.size == 0 or scores_inds_keep.size == 0: # if no box retained, keep the original one
            keep_fusion_boxes.append(dets[i])
            if DEBUG:
                print("keep original det")
            continue

        inds_fusion = np.intersect1d(IOU_inds_keep, scores_inds_keep)
        if DEBUG:
            if inds_fusion.size>1:
                print("boxes for fusion:", inds_fusion)
                print(dets[order[inds_fusion]])

        x1_fusion = x1[order[inds_fusion]]
        y1_fusion = y1[order[inds_fusion]]
        x2_fusion = x2[order[inds_fusion]]
        y2_fusion = y2[order[inds_fusion]]
        scores_fusion = scores[order[inds_fusion]]
        fusion_box = np.zeros((1,5))
        fusion_box[0][0] = np.sum(x1_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[0][1] = np.sum(y1_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[0][2] = np.sum(x2_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[0][3] = np.sum(y2_fusion * scores_fusion) / np.sum(scores_fusion)
        fusion_box[0][4] = scores_fusion[0]
        if DEBUG:
            print("fusion_box:", fusion_box)

        keep_fusion_boxes.append(fusion_box)

        # boxes with small overlap are kept for another loop
        inds_next = np.where(ovr <= IOU_thresh)[0]
        order = order[inds_next]

    keep_fusion_boxes = np.array(keep_fusion_boxes)

    return keep_fusion_boxes
