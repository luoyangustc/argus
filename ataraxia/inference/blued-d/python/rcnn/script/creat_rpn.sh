#!/usr/bin/env bash
# 网络模型：resnet训练的第2轮epoch的结果
# 数据集：ILSVRC 2017 分类的val数据集
LOG=proposal_inceptionv3.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m rcnn.tools.proposal --network inceptionv3                        \
                                  --prefix model/inceptionv3rpn          \
				  --image_set 2007_trainval     \
                                  --gpu 2                                   \
                                  --epoch 0                                 \
                                  >${LOG} 2>&1 &
                                 # --dataset imagenet           \
                                 # --image_set trainall                           \
