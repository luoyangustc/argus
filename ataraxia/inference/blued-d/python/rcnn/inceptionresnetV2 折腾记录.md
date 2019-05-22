# Inception Resnet v2 折腾记录
## Inception Resnet v2 模型来源
ImageNet 上分类用的模型结构请参考:
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)
预训练的模型来源为
[caffemodel](https://pan.baidu.com/s/1jHPJCX4#list/path=%2F)
### caffe 模型到 mxnet 模型转换
1. 使用 mxnet tools 下的 caffe convet 工具
2. 使用时先安装 `pycaffe` 然后进入工具目录下 执行 `make` 最后调用运行命令

	> python convert_model.py xx.prototxt xx.caffemodel outputname

3. 原来的工具不支持 1X7 或者 7X1 的 kernelsize 的模型，需要修改之后才能使用，具体修改的代码可以参考 [PR](https://github.com/dmlc/mxnet/pull/6139) 

### 一些折腾过的参数
* 综述： 现有的简单的将inception Resnet V2 模型用来当作预训练模型 `FineTune` 得到的检测模型并不是很好，下面列举一些尝试过的改进方法。
* 鉴于收敛速度比较慢，尝试过调整 `Dropout` 参数: 从0.2 调整至 0.5 最后到 去掉，发现收敛速度并没有明显的变化，收敛的情况也没有明显的变化，显然这个参数并不是决定最终的结果
* 调整学习率： 原来设置的 `0.001` 的学习率在inception Resnet v2 `FineTune` 的过程中并不适用，会直接导致模型分散，实验发现，`0.0001` 的学习率下模型收敛的比较好，再调小的话会导致收敛过慢
* MxNet 中 `BatchNorm` 层中 `eps` 参数默认为 `2e-5` 发现与转换时候的 `Prototxt` 中设置的`0.01` 不符,	尝试修改该参数： 修改后发现整体并没有明显的变化，显然也不是这个原因导致效果不好
* Inception Resnet v2 ： 模型在不断的下采样的过程中使用的是 `kernel_size =3 stride=2`的卷积层，导致在图片宽高为奇数时候会使得图片不是除以二缩小，参考resnet中的设计，添加 `pad=1` 参数可以修复这个问题。对比实验表明，这样的修改可以大大的减少第一个batch的误差，但是并没有彻底解决收敛不好的问题。 
* 调整固定的层数： 参考 `vggnet` 和 `resnet` 保留的是前两个下采样层的结果，之前保留的也是`conv1` `conv2` 和 `conv3` 的结果，调整过这个参数，增加或者减少固定的层数，均会导致第一个batch的误差大幅度增加。
* 调整 Pixel_Mean 设置：之前配置 参考的是 `resnet` 并不减去图像均值。而 `prototxt` 中是有图片的均值存在的，所以将这个问题修改回来，但是对比实验发现，并不影响最后的一个收敛情况，看了这个也不是导致收敛不好的关键因素
* ave pooling size 问题： 这个问题还没有机会进行对比实验。 `prototxt` 文件 输入的size 是`331X331` 计算到最后的pooling层就是 9x9 而标准的 `17X17` 最终的是 8x8 ，这个问题改怎么解决，欢迎大家讨论。
*  17X17 接入地方问题： inception Resnet v2 中在 `17X17` 会重复 20 次 inception layer ，之前是在ROI pooling 之后做的，由于 Pooling 后的框比较多，所以耗内存较大，修改之后，放在ROI pooling 之前完成这个20次，可以加少内存消耗和计算量，通过对比实验发现，这个改动有助于减少一开始的误差，加速训练。
*  随着 `epoch` 的增加，所有的loss 都会缓慢减少，几个 epoch 测试的结果也会有提高，延长训练时间会有提高，但是训练非常缓慢，这个是目前要解决的最大的问题 。
*  可能的一些解决办法： 单独训练RPN 再单独训练rcnn 是否可以加快训练，网络层数太深，减少一些层数是不是可以加快训练，voc 数据太小不适合网络，换一个大一点的数据是不是可以加快训练



