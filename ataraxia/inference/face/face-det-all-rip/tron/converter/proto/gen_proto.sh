#!/bin/bash
##############################################################################
# Example command to generate python protobuf files with shell.
##############################################################################
# 
# This script shows how one can generate python protobuf files with shell.


PROTO_ROOT="$(cd "$(dirname "$0")/" && pwd)"
FILE_NAME=caffe.proto

wget http://oxmz2ax9v.bkt.clouddn.com/$FILE_NAME -O $PROTO_ROOT/$FILE_NAME

###################### Generate Protobuf #######################
protoc --python_out=$PROTO_ROOT \
       --proto_path=$PROTO_ROOT \
       $PROTO_ROOT/*.proto
