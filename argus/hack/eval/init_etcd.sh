#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
export ETCDCTL_API=3
etcdctl del --prefix /ava/serving

etcdctl put /ava/serving/sts/hosts '[{"key":"1", "host":"127.0.0.1:5555"}]'
etcdctl put /ava/serving/nsq/consumer '[{"addresses":["127.0.0.1:4161"]}]'
etcdctl put /ava/serving/nsq/producer '["127.0.0.1:4150"]'

etcdctl put /ava/serving/logpush '{"open":true,"url":"http://atflowlogproxy.k8s-xs.qiniu.io/v1/dataflows/logs","timeout_second":3}'

etcdctl put /ava/serving/worker/default '{
  "timeout": 10000000000,
  "max_concurrent": 20,
  "delay4batch": 100000000
}'

etcdctl put /ava/serving/app/default/metadata '{
  "public": true
}'

etcdctl put /ava/serving/app/metadata/hello-eval '{
  "batch_size": 10
}'

etcdctl put /ava/serving/app/release/hello-eval/1 '{
  "batch_size": 10,
  "image_width": 224,
  "tar_file": "http://q.hi-hi.cn/ava-caffe-model-classify-res/ava-caffe-model-classify.tar"
}'

etcdctl put /ava/argus/video/vframe '{
  "mode": 0,
	"interval": 5
}'

etcdctl put /ava/argus/video/segment '{
  "mode": 1,
	"interval": 5
}'

etcdctl put /ava/argus/video/jobs '{
  "mgo_host": "127.0.0.1:27017"
}'

etcdctl put /ava/argus/video/worker '{
  "pool_max": 10
}'

etcdctl put /ava/argus/video/op/config/foo '{
  "host": "127.0.0.1:1234",
	"timeout": 1000000000,
	"params": {}
}'

etcdctl put /ava/argus/gate/routes '{}'


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

etcdctl get --prefix ''
