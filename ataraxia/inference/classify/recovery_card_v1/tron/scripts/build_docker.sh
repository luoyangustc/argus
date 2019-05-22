#!/bin/bash
##############################################################################
# Example command to build the whole project with docker.
##############################################################################
# 
# This script shows how one can build tron with docker.

TRON_ROOT="$(cd "$(dirname "$0")/../" && pwd)"
TAG_NAME="reg-xs.qiniu.io/atlab/terror-pulp-48-20180521:v1"

cd $TRON_ROOT
docker build -t $TAG_NAME -f docker/Dockerfile .
