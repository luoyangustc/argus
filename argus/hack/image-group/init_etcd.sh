#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
export ETCDCTL_API=3

etcdctl put /ava/serving/sts/hosts '[{"key":"1", "host":"10.200.30.13:5555"},{"key":"2", "host":"10.200.30.14:5555"},{"key":"3", "host":"10.200.30.15:5555"}]'

etcdctl put /ava/argus/image_group/cache '{"chunk_size":67108864,"size":268435456,"slabs":[{"count":8,"size":4194304},{"count":4,"size":16777216},{"count":2,"size":67108864}]}'
etcdctl put /ava/argus/image_group/worker '{"gogc":100}'

etcdctl put /ava/argus/image_group/current_feature '"v1"'
etcdctl put /ava/argus/image_group/feature/v1 '{"feature_host": "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001", "feature_version": "", "feature_length": 4096, "feature_byteorder": 0, "reserve_byteorder": true, "threshold": 0.4}'

etcdctl put /ava/argus/image_group/fetch '{"url":"http://localhost:8201/v1/image/group/foo/feature"}'

etcdctl put /ava/argus/image_group/search '{
  "saver": {
    "kodo": {
      "bucket": "argus-image-group",
      "config": {
        "AccessKey": "cThFVgwOXUfcpe3QWsovRn4xG8tSF20KTT6QOoPG",
        "SecretKey": "gE_GKmqhbiC89Lyghtu6r5fU1_Xf0x_7dEMdCDl_",
        "RSHost": "http://10.200.20.25:12501",
        "RSFHost": "http://10.200.20.25:12501",
        "APIHost": "http://10.200.20.23:12500",
        "UpHosts": ["http://10.200.20.23:5010"],
        "IoHost": "http://10.200.20.23:5000"
      },
      "prefix": "v1",
      "zone": 0,
      "uid": 1380538984
    }
  }
}'

etcdctl put /ava/argus/image_group/mgo '{
    "host": "localhost",
    "db": "argus_ig",
    "mode": "strong",
    "timeout": 3
}'

etcdctl get --prefix ''
