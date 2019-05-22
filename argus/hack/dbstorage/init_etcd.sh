#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
export ETCDCTL_API=3

etcdctl put /ava/argus/dbstorage/thread_num '10'
etcdctl put /ava/argus/dbstorage/max_run_task '3'

etcdctl put /ava/argus/dbstorage/feature_group '{"host": "http://localhost:7201", "timeout_in_second": 0}'
etcdctl put /ava/argus/dbstorage/serving '{"host": "http://ava-serving-gate.cs.cg.dora-internal.qiniu.io:5001", "timeout_in_second": 0}'

etcdctl put /ava/argus/dbstorage/mgo '{
    "host": "localhost",
    "db": "dbstorage",
    "mode": "strong",
    "timeout": 3
}'

etcdctl get --prefix ''
