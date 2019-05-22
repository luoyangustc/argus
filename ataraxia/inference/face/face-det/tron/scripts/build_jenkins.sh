#!/bin/bash
# usage: sh scripts/build_jenkins.sh $AVAPUBLIC_USERNAME $AVAPUBLIC_PASSWORD $REGISTRY_NAME $IMAGE_NAME

set -e
set -x

hostname

docker build -t $4 -f docker/Dockerfile .

docker login -u $1 -p $2 $3

docker push $4

echo "----------" build success $4 "----------"
