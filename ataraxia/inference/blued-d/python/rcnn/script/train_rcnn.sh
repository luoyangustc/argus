#!/usr/bin/env bash

# run this experiment with
# nohup bash script/resnet_voc00712.sh 0,1 &> resnet_voc0712.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to resnet_voc0712.log
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python -m rcnn.tools.train_rcnn --network resnet --dataset imagenet --image_set train --pretrained model/resnet-101-all --pretrained_epoch 10 --use_global_context --use_roi_align --use_data_augmentation --gpus 0,1,2,3 #$1