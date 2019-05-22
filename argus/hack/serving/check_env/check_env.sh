#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

which etcd > /dev/null || (echo "!!! etcd not install" && exit -1)
which nsqd > /dev/null || (echo "!!! nsq not install" && exit -1)

