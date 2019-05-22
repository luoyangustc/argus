# -*- coding: utf-8 -*-

import time
import os.path as osp
import json
import struct
import traceback

from evals.utils import CTX

import numpy as np
from numpy.linalg import norm

from scipy.spatial.distance import cosine as ssd_cosine_dist
from sklearn import cluster


def _compute_pairwise_distance(ft1, ft2):
    try:
        dist = 1.0 - np.dot(np.array(ft1['feature']), np.array(ft2['feature']))
        if dist < 0.0:
            dist = 0.0
    except:
        CTX.logger.error('_compute_pairwise_distance() failed')
        traceback.print_exc()
        return None
    return dist


def _calc_group_centers(clustered_ft_list, max_gid=-1):
    try:

        if len(clustered_ft_list) < 1 or max_gid < 0:
            CTX.logger.info('len(clustered_ft_list) < 1 or max_gid < 0')
            return None

        ft_vec_len = len(clustered_ft_list[0]['feature'])

        grp_len = max_gid + 1
        grp_cnts = [0 for i in range(grp_len)]
        grp_centers = [{'feature': np.zeros((ft_vec_len,))}
                       for i in range(grp_len)]

        for ft in clustered_ft_list:
            if ft['group_id'] > -1 and ((ft['gt_id'] == -1) or (ft['gt_id'] == ft['group_id'])):
                grp_centers[ft['group_id']
                            ]['feature'] += np.array(ft['feature'])
                grp_cnts[ft['group_id']] += 1

        for i in range(grp_len):
            if grp_cnts[i] > 0:
                factor = 1.0 / grp_cnts[i]
                grp_centers[i]['feature'] *= factor
            # use default center [0, 0, ...]
            # else:
            #     grp_centers[i]['feature'] = None
    except:
        CTX.logger.error('_calc_group_centers() failed')
        traceback.print_exc()
        return None

    return grp_centers


def _vec_compute_distance_matrix(ft_list):
    try:
        dist_matrix = 1.0 - np.dot(ft_list, np.transpose(ft_list))
    except:
        CTX.logger.error('_vec_compute_distance_matrix() failed')
        traceback.print_exc()
        return None

    return dist_matrix


def _compute_distance_matrix(ft_list):
    try:
        nsamples = len(ft_list)
        if not (nsamples > 0):
            CTX.logger.info('must have len(ft_list)>0')

        fts = [ft['feature'] for ft in ft_list]
        fts_mat = np.array(fts)

        dist_matrix = _vec_compute_distance_matrix(fts_mat)
    except:
        CTX.logger.error('_compute_distance_matrix() failed')
        traceback.print_exc()
        return None

    return dist_matrix


def _cluster_features(dist_matrix, dist_thresh=0.5, min_samples=2):
    try:
        cluster_estimator = cluster.DBSCAN(
            metric='precomputed', eps=dist_thresh, min_samples=min_samples)
        cluster_estimator.fit(dist_matrix)
        y_pred = cluster_estimator.labels_.astype(np.int)
    except:
        CTX.logger.error('_cluster_features() failed')
        traceback.print_exc()
        return None
    return y_pred


def cluster_all_features(ft_list, dist_thresh=0.5, min_samples=2, max_gid=-1):
    try:
        if len(ft_list) < min_samples:
            for ft in ft_list:
                ft['group_id'] = -1
                ft['distance_to_center'] = 0.0

            return ft_list

        t0 = time.time()

        dist_matrix = _compute_distance_matrix(ft_list)
        # print "dist_matrix: ", dist_matrix

        t1 = time.time()
        t = t1 - t0

        # print "ft_list: ", ft_list

        CTX.logger.info('===>facex-cluster: _compute_distance_matrix() for'
                        ' %d features takes: %f seconds' % (len(ft_list), t))

        t0 = time.time()

        labels = _cluster_features(dist_matrix, dist_thresh, min_samples)

        t1 = time.time()
        t = t1 - t0

        CTX.logger.info('_cluster_features() returns: ' + str(labels))

        CTX.logger.info('===>facex-cluster: _cluster_features() for'
                        ' %d features takes: %f seconds' % (len(ft_list), t))

        gid_base = max_gid + 1

        for (ft, label) in zip(ft_list, labels):
            ft['group_id'] = (label + gid_base) if (label > -1) else -1

        max_gid = labels.max() + gid_base

        clustered_ft_list = []
        single_ft_list = []

        grp_centers = []
        if max_gid >= 0:
            grp_centers = _calc_group_centers(ft_list, max_gid)
        # print "grp_centers", grp_centers

        for ft in ft_list:
            if ft['group_id'] > -1:
                ft['distance_to_center'] = _compute_pairwise_distance(
                    ft, grp_centers[ft['group_id']])
                clustered_ft_list.append(ft)
            else:
                ft['distance_to_center'] = 0.0
                single_ft_list.append(ft)
    except:
        CTX.logger.error('cluster_all_features() failed')

        traceback.print_exc()
        return None
    # return results
    return (clustered_ft_list, single_ft_list)


def _find_min_dist_grp_id(new_ft, grp_centers):
    try:
        if len(grp_centers) < 1:
            return (-1, None)
        elif len(grp_centers) == 1:
            best_grp_id = 0
            min_dist = _compute_pairwise_distance(new_ft, grp_centers[0])
        else:
            grp_centers_list = [ct['feature'] for ct in grp_centers]
            grp_centers_mat = np.array(grp_centers_list)
            dist_mat = 1.0 - np.dot(grp_centers_mat,
                                    np.array(new_ft['feature']))

            best_grp_id = dist_mat.argmin()
            min_dist = dist_mat[best_grp_id]
    except:
        CTX.logger.error('_find_min_dist_grp_id() failed')
        traceback.print_exc()
        return None

    return (best_grp_id, min_dist)


def incrementally_cluster(clustered_ft_list, single_ft_list, new_ft_list, dist_thresh=0.5, min_samples=2, max_gid=-1):
    try:
        if (len(single_ft_list) + len(new_ft_list) < min_samples
                or len(new_ft_list) == 0 or len(clustered_ft_list) == 0):
            for item in clustered_ft_list:
                item['distance_to_center'] = 0.0
            single_ft_list = single_ft_list + new_ft_list
            for item in single_ft_list:
                item['group_id'] = -1
                item['distance_to_center'] = 0.0
            return (clustered_ft_list, single_ft_list)

        if max_gid < 0:
            max_gid = 0
            for ft in clustered_ft_list:
                if max_gid < ft['group_id']:
                    max_gid = ft['group_id']

        t0 = time.time()
        grp_centers = _calc_group_centers(clustered_ft_list, max_gid)
        new_clustered_flag = False

        for ft in new_ft_list:
            (best_grp_id, min_dist) = _find_min_dist_grp_id(ft, grp_centers)

            if best_grp_id >= 0 and min_dist < dist_thresh:
                ft['group_id'] = best_grp_id
                clustered_ft_list.append(ft)
                new_clustered_flag = True
            else:
                ft['group_id'] = -1
                single_ft_list.append(ft)

        if new_clustered_flag:
            grp_centers = _calc_group_centers(clustered_ft_list, max_gid)

        for ft in clustered_ft_list:
            if ft['group_id'] > -1:
                ft['distance_to_center'] = _compute_pairwise_distance(
                    ft, grp_centers[ft['group_id']])
            else:
                ft['distance_to_center'] = 0.0

        t1 = time.time()
        t = t1 - t0
        CTX.logger.info('===>facex-cluster: compairing %d new ft with'
                        ' %d group centers takes: %f seconds' % (len(new_ft_list), len(grp_centers), t))

        if len(single_ft_list) > min_samples:
            (clustered_ft_list2, single_ft_list2) = cluster_all_features(
                single_ft_list, dist_thresh, min_samples, max_gid)
            clustered_ft_list.extend(clustered_ft_list2)
            single_ft_list = single_ft_list2
    except:
        CTX.logger.error('incrementally_cluster() failed')

        traceback.print_exc()
        return None
    return (clustered_ft_list, single_ft_list)
