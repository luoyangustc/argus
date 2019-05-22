#!/usr/bin/env bash

# run this experiment with
# nohup bash script/resnet_imagenet.sh 0,1 &> resnet_imagenet.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to resnet_voc07.log
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

#python train_end2end.py --network resnet --dataset imagenet --gpu $1
python train_alternate.py --network resnet --dataset imagenet --gpu $1 --use_global_context --use_data_augmentation  --use_roi_align
python test.py --network resnet --gpu $gpu --use_global_context --use_roi_align
