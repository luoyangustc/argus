#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''search facex
'''

import json
import numpy
import struct

from evals import utils
from evals.utils import *
from evals.utils.error import *
from evals.utils.logger import logger


def short_side(pts):
    
    if type(pts) is not list or len(pts)!=2 and len(pts)!=4:
        return 0, "invalid pts"
    for pt in pts:
        if type(pt) is not list or len(pt)!=2:
            return 0, "invalid pts"

    if len(pts)==4:
        short = pts[2][0]-pts[0][0]
        if pts[2][0]-pts[0][0] > pts[2][1]-pts[0][1]:
            short=pts[2][1]-pts[0][1]
    elif len(pts)==2:
        short = pts[1][0]-pts[0][0]
        if pts[1][0]-pts[0][0]>pts[1][1]-pts[0][1]:
            short=pts[1][1]-pts[0][1]
    return short, ""


@create_net_handler
def create_net(configs):
    '''
        net init
    '''

    logger.info("[Python net_init] load configs: %s", configs, extra={"reqid": ""})

    file_features = configs.get('custom_files', {}).get('features.line', None)
    file_labels = configs.get('custom_files', {}).get('labels.line', None)
    if file_features is None or file_labels is None:
        logger.error("need file_features/file_labels",extra={"reqid": ""})
        return {}, 400 ,"miss some custom file"

    custom_values=configs.get('custom_params',{})

    with open(file_features, 'r') as _f:
        # {'index': 1, 'url': '', 'pts':[], 'feature':[]}
        features = [json.loads(line) for line in _f.readlines()]
    
    large_features = [ line for line in features if line.get("size","")=="large"]
    small_features = [ line for line in features if line.get("size","")=="small"]
    xsmall_features = [ line for line in features if line.get("size","")=="XSmall"]

    with open(file_labels, 'r') as _f2:
        labels = [line.strip() for line in _f2.readlines()]
    
    thresholds=custom_values.get('threshold', [[0.38,0.4,0.42],[0.38,0.4,0.42],[0.35,0.375,0.4]])
    size_limits=custom_values.get("size_limit",[24,32,60])

    if len(labels)==0:
        return {}, 400 ,"invliad numbers of labels "
    if len(large_features)==0 or len(small_features)==0 or len(xsmall_features)==0:
        return {}, 400 ,"invliad numbers of base datasets "
    if len(thresholds)!=3 or len(thresholds[0])!=3 or len(thresholds[1])!=3 or len(thresholds[2])!=3:
         return {}, 400 ,"invliad numbers of thresholds, should be array[3][3]"
    if len(size_limits) !=3:
        return {}, 400 ,"invliad numbers of subset size_limit , should be array[3]"

    cu_large_features = numpy.array([feature.get('feature',[]) for feature in large_features])
    cu_small_features = numpy.array([feature.get('feature',[]) for feature in small_features])
    cu_xsmall_features = numpy.array([feature.get('feature',[]) for feature in xsmall_features])

    return {
             'large_features': large_features, 
             'small_features':small_features,
             'xsmall_features':xsmall_features,
             'labels': labels,
            
             'cm_large_features': cu_large_features,
             'cm_small_features': cu_small_features,
             'cm_xsmall_features': cu_xsmall_features,
             'thresholds': thresholds,
             'sizelimits':size_limits
           }, 0 , ''


@net_inference_handler
def net_inference(model, args):
    '''
        net inference
    '''
    large_features = model.get('large_features', [])
    small_features = model.get('small_features', [])
    xsmall_features = model.get('xsmall_features', [])
    cm_large_features = model.get('cm_large_features',None)
    cm_small_features = model.get('cm_small_features',None)
    cm_xsmall_features = model.get('cm_xsmall_features',None)
    labels = model.get('labels', [])
    thresholds=model.get('thresholds', 0.0)
    size_limits=model.get('sizelimits')
    buf=None
    limit = 1
    req = args[0]
    pts = req.get('data', {}).get('attribute',{}).get('pts',[])

    logger.info("[Python inference]: limit:%s,threshold:%s", limit,thresholds, extra={"reqid": ""})

    features=large_features
    cm_features=cm_large_features
    threshold=thresholds[2]
    if len(pts)!=0:
        srt,err =short_side(pts)
        logger.info("[Python inference]: shortest side %d, err: %s", srt,err,extra={"reqid": ""})
        if err != "":
            return {}, 400 , err
        if srt < size_limits[0]:
            return [{'code': 0, 'message': 'pts should larger than {}x{}'.format(size_limits[0],size_limits[0]),'result':{"confidences": []}}],0,'' 
        elif srt <  size_limits[1]:   
            features=xsmall_features
            cm_features=cm_xsmall_features
            threshold=thresholds[0]
        elif srt <  size_limits[2]:
            features=small_features
            cm_features=cm_small_features
            threshold=thresholds[1]
    #阈值映射
    score_map= lambda b,m,u,f,s: b+m*(s-u)/f

    try:
        if os.path.exists(req.get('data', {}).get('uri', '')):
            _f=open(req.get('data', {}).get('uri', ''), 'rb')
            buf = _f.read()
            _f.close()
        elif req.get('data', {}).get('body', '') is not None:
            buf=req.get('data', {}).get('body', '')
        else:
            return {},400,"failed to get feature data"
        feature = struct.unpack(">"+ str(len(buf)/4) + 'f', buf)
    except struct.error as _e:
        return {}, 400 , _e.message

    if feature is None:
        logger.error('read input failed.',extra={"reqid": ""})
        return {}, 500, "read featue failed"

    if len(feature) != (len(features[0].get('feature', [])) if len(features) > 0 else 4096):
        logger.error("feature len %d %d", len(feature), len(features[0].get('feature', [])),extra={"reqid": ""})
        return {}, 400, "bad feature input"

    target_feature =numpy.array(feature)
    cosins =cm_features.dot(target_feature)
    
    if "params" in req and "limit" in req["params"]:
            if (type(req["params"]["limit"]) is int or req["params"]["limit"].isdigit()) and \
                int(req["params"]["limit"]) <= len(labels) and int(req["params"]["limit"]) > 0:
                limit = int(req["params"]["limit"])
    
    cosins=cosins.squeeze() 
    indexs = cosins.argsort()[::-1][:limit]          
    
    ret =[{'code': 0, 'message': '','result':{"confidences": []}}]
    for i in xrange(len(indexs)):
        score=float(cosins[indexs[i]])
        if score < threshold[1] :
            logger.info("ret {}".format(ret),extra={"reqid": ""})
            return ret,0,None
        #score map，四个区段分别映射
        #if score <threshold[0]:
        #    score=score_map(0,thresholds[2][0],score,threshold[0],threshold[0])
        if score > threshold[2]:
            score=score_map(thresholds[2][2],1-thresholds[2][2],threshold[2],1-threshold[2],score)
        else:
            #score >= threshold[1] and score <= threshold[2]
            score=score_map(thresholds[2][1],thresholds[2][2]-thresholds[2][1],threshold[1],threshold[2]-threshold[1],score)

        ret[0]["result"]["confidences"].append({
                'index': features[indexs[i]].get('index', 0),
                'class': labels[features[indexs[i]].get('index', 0)],
                'group': features[indexs[i]].get('group', ''),
                'score': score,
                'sample': {
                    'url': features[indexs[i]].get('url', ''),
                    'pts': features[indexs[i]].get('pts', ''),
                    'id': features[indexs[i]].get('id', ''),
                }
            })
    return ret ,0 , None
