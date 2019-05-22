#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
export ETCDCTL_API=3

etcdctl put /ava/serving/nsq/consumer '[{"addresses":["127.0.0.1:4161"]}]'
etcdctl put /ava/serving/nsq/producer '["127.0.0.1:4150"]'

etcdctl put /ava/argus/jobs/mq '{
    "idle_job_timeout": 300000000000,
    "mgo_pool_limit": 50,
    "mgo": {
        "host": "localhost",
        "db": "tuso_dev_local",
        "mode": "strong",
        "timeout": 3
    }
}'

etcdctl put /ava/argus/jobs/cmds/tuso-search '{"master":{"pool_max":1}}'
etcdctl put /ava/argus/jobs/cmds/tuso-search/config '{"batch_size":512}'

etcdctl put /ava/argus/tuso/jobgate_api '{
    "host": "http://localhost:9205",
    "timeout_second": 5
}'

etcdctl put /ava/argus/tuso/internal_api '{
    "host": "http://localhost:9204",
    "timeout_second": 5
}'

etcdctl put /ava/argus/tuso/mgo '{
    "host": "localhost",
    "db": "tuso_dev_local",
    "mode": "strong",
    "timeout": 3
}'

etcdctl put /ava/argus/tuso/kodo '{
    "ak": "DoDz1cdD6AbXLzY6ss_dT5VWvdmDzg0KhSikgTqI",
    "sk": "ic6Jwb5RNyskxHeN_vEKBkGNRAuqpfhuaXolB602",
    "bucket": "tuso-feature",
    "region": 0,
    "domain": "p4r2vw4xc.test.bkt.clouddn.com",
    "prefix": "test1",
    "io_host": ["http://10.200.20.23"],
    "up_host": ["http://10.200.20.23:5010"]
}'

etcdctl put /ava/argus/tuso/image_feature_api '{
    "host": "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001",
    "timeout_second": 5
}'

etcdctl put /ava/argus/tuso/image_feature_concurrency_num 5

etcdctl get --prefix ''
