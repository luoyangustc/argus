#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
export ETCDCTL_API=3

etcdctl put /ava/serving/sts/hosts '[{"key":"1", "host":"10.200.30.13:5555"},{"key":"2", "host":"10.200.30.14:5555"},{"key":"3", "host":"10.200.30.15:5555"}]'

etcdctl put /ava/argus/face_group/cache '{"chunk_size":67108864,"size":268435456,"slabs":[{"count":8,"size":4194304},{"count":4,"size":16777216},{"count":2,"size":67108864}]}'

etcdctl put /ava/argus/face_group/current_feature '"v2"'
etcdctl put /ava/argus/face_group/feature/v1 '{"detect_host": "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001", "feature_host": "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001", "feature_version": "v2", "feature_length": 512, "feature_byteorder": 1, "reserve_byteorder": true, "threshold": 0.525}'
etcdctl put /ava/argus/face_group/feature/v2 '{"detect_host": "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001", "feature_host": "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001", "feature_version": "v3", "feature_length": 512, "feature_byteorder": 1, "reserve_byteorder": false, "threshold": 0.4}'

etcdctl put /ava/argus/face_group/fetch '{"url":"http://localhost:7201/v1/face/group/foo/feature"}'

etcdctl put /ava/argus/face_group/search '{"saver":{"kodo":{"bucket":"argus-face-group","config":{"AccessKey":"cThFVgwOXUfcpe3QWsovRn4xG8tSF20KTT6QOoPG","SecretKey":"gE_GKmqhbiC89Lyghtu6r5fU1_Xf0x_7dEMdCDl_","RSHost":"http://10.200.20.25:12501","RSFHost":"http://10.200.20.25:12501","APIHost":"http://10.200.20.23:12500","UpHosts":["http://10.200.20.23:5010"],"IoHost":"http://10.200.20.23:5000"},"prefix":"v1","zone":0,"uid":1380538984}}}'

etcdctl put /ava/argus/face_group/mgo '{
    "host": "localhost",
    "db": "argus",
    "mode": "strong",
    "timeout": 3
}'

etcdctl get --prefix ''
