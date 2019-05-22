#!/usr/bin/env bash

# run this experiment with
# nohup bash script/resnet_imagenet.sh 0,1 &> resnet_imagenet.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to resnet_voc07.log
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0

python demo_for_imglist.py --network resnet --dataset imagenet --image_set test --prefix model/resnet152 --epoch 10 --gpu $gpu # --use_global_context --use_roi_align --use_box_voting
