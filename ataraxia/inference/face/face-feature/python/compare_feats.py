#! /usr/bin/env python

import os
import sys
import os.path as osp

import numpy as np
from numpy.linalg import norm

#from matio import load_mat


def load_npy(npy_file):
    mat = None
    if osp.exists(npy_file):
        mat = np.load(npy_file)
    else:
        err_info = 'Can not find file: ' + npy_file
        raise Exception(err_info)

    return mat


def load_feat(feat_file, flatten=True):
    feat = None
    if feat_file.endswith('npy'):
        feat = load_npy(feat_file)
    # elif feat_file.endswith('bin'):
    #     feat = load_mat(feat_file)
    else:
        raise Exception(
            'Unsupported feature file. Only support .npy and .bin (OpenCV Mat file)')

    if flatten:
        feat = feat.flatten()

    return feat


def calc_similarity_cosine(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)
    print 'feat1_norm: ', feat1_norm
    print 'feat2_norm: ', feat2_norm

    sim = np.dot(feat1, feat2.T) / (feat1_norm * feat2_norm)

    return sim


def calc_similarity_L2(feat1, feat2):
    diff_vec = feat1 - feat2
    diff_vec_norm = norm(diff_vec)
#    print 'diff_vec_norm: ', diff_vec_norm

    L2_sim = - np.dot(diff_vec, diff_vec.T)

    return L2_sim


def calc_similarity_norm_L2(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)
#    print 'feat1_norm: ', feat1_norm
#    print 'feat2_norm: ', feat2_norm

    norm_diff_vec = feat1 / feat1_norm - feat2 / feat2_norm
    norm_diff_vec_norm = norm(norm_diff_vec)
#    print 'norm_diff_vec_norm: ', norm_diff_vec_norm

    L2_sim = - np.dot(norm_diff_vec, norm_diff_vec)

    return L2_sim


def calc_similarity(feat1, feat2, sim_type='cosine'):
    sim = 0
    if sim_type is 'cosine':
        sim = calc_similarity_cosine(feat1, feat2)
    elif sim_type is 'L2':
        sim = calc_similarity_norm_L2(feat1, feat2)
    elif sim_type is 'norm_L2':
        sim = calc_similarity_norm_L2(feat1, feat2)
    else:
        err_info = 'sim_type must be one of {"cosine", "L2", "norm_L2"}!'
        raise Exception(err_info)

    return sim


def compare_feats(file1, file2):
    print '\n======================'
    print '---> Load feat file 1: ', file1
    ft1 = load_feat(file1)
    if ft1 is None:
        print "Failed to load feature1's .npy"
        return None

    print '---> Load feat file 2: ', file2
    ft2 = load_feat(file2)
    if ft2 is None:
        print "Failed to load feature2's .npy"
        return None

    sim = calc_similarity_cosine(ft1, ft2)
    print '--->cosine similarity: ', sim

    L2_sim = calc_similarity_L2(ft1, ft2)
    print '--->L2 similarity: ', L2_sim

    L2_sim_norm = L2_sim / (norm(ft1) * norm(ft2))
    print '--->L2_sim_norm: ', L2_sim_norm

    norm_L2_sim = calc_similarity_norm_L2(ft1, ft2)
    print '--->norm_L2_sim: ', norm_L2_sim

    return sim


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        file1 = 'rlt_features/face_chip_1_1.npy'
        file2 = 'rlt_features/face_chip_1_2.npy'

    compare_feats(file1, file2)
