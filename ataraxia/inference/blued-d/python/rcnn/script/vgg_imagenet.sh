#!/usr/bin/env bash

# run this experiment with
# nohup bash script/vgg_voc07.sh 0,1 &> vgg_voc07.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to vgg_voc07.log
gpu=${1:0:1}
#echo $2
if [ -n "$2" ];then 
  epoch=$2
else
  epoch=0
fi

#echo $gpu
#echo $epoch
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

if [ $epoch = 0 ];then 
	python train_end2end.py --dataset imagenet --gpu $1 
else
	python train_end2end.py --dataset imagenet --gpu $1 --lr 0.0001 --prefix model/e2e --resume  --begin_epoch $epoch --end_epoch 20 --lr_step 15
fi

# python test.py --gpu $gpu
