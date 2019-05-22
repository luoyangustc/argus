#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
export ETCDCTL_API=3
etcdctl del --prefix /ava

etcdctl put /ava/serving/sts/hosts '[{"key":"1", "host":"127.0.0.1:5555"}]'
etcdctl put /ava/serving/nsq/consumer '[{"addresses":["127.0.0.1:4161"]}]'
etcdctl put /ava/serving/nsq/producer '["127.0.0.1:4150"]'

etcdctl put /ava/argus/jobs/mq '{
    "idle_job_timeout": 300000000000,
    "mgo_pool_limit": 50,
    "mgo": {
        "host": "localhost",
        "db": "argus_job_local",
        "mode": "strong",
        "timeout": 3
    }
}'

etcdctl put /ava/argus/video/vframe '{
  "mode": 0,
	"interval": 5
}'

etcdctl put /ava/argus/video/segment '{
  "mode": 1,
	"interval": 5
}'

etcdctl put /ava/argus/video/op/config/foo '{
  "host": "127.0.0.1:1234",
	"timeout": 1000000000,
	"params": {}
}'

etcdctl put /ava/argus/jobs/workers/inference-image '{
  "worker": {
    "max_in_flight": 100
  },
  "URL":"http://ava-argus-gate.xs.cg.dora-internal.qiniu.io:5001/v1/image/censor"
}'

etcdctl put /ava/argus/jobs/workers/inference-video '{
  "worker": {
    "max_in_flight": 3
  }
}'

etcdctl put /ava/argus/jobs/cmds/bucket-censor '{
  "master": {
      "pool_max": 1,
      "concurrent_pre_job": 800,
      "topic_default": "first_argus_jobw_inference-image",
      "topic_options": {
        "image": "first_argus_jobw_inference-image",
        "video": "first_argus_jobw_inference-video"
      }
  }
}'

etcdctl put /ava/argus/jobs/cmds/bucket-censor/scan '{
  "qconf": {},
  "kodo": {
      "UpHosts": [
          "http://nbxs-gate-up.qiniu.com"
      ],
      "APIHost": "http://api.qiniu.com"
  },
  "hack": {
    "ak": "",
    "sk": ""
  }
}'

etcdctl get --prefix ''
