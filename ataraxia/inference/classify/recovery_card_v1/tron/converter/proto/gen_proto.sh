#!/bin/bash
##############################################################################
# Example command to generate python protobuf files with shell.
##############################################################################
# 
# This script shows how one can generate python protobuf files with shell.


PROTO_ROOT="$(cd "$(dirname "$0")/" && pwd)"

###################### Generate Protobuf #######################
protoc --python_out=$PROTO_ROOT \
       --proto_path=$PROTO_ROOT \
       $PROTO_ROOT/tron.proto

protoc --python_out=$PROTO_ROOT \
       --proto_path=$PROTO_ROOT \
       $PROTO_ROOT/caffe.proto
