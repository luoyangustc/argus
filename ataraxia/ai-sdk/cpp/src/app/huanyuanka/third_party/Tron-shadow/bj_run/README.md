# Atlab SDK TEAM Inference LIB 网信办5合1网络+人脸ONET对齐网络+人脸提特征网络

本项目主要为网信办五合一网络的TensorRT加速版本，Qiniu.cf平台上有在P4卡上的[测试报告](https://cf.qiniu.io/pages/viewpage.action?pageId=91328776)

## 推理
### 1、编译文件
```Shell
mkdir build
cd build
cmake ..
make
```


### 2、准备测试数据（默认执行路径在 build 下，所以请将文件解压至 build 目录下）

下载[网络模型](http://pbv7wun2s.bkt.clouddn.com/bjrun-caffe-model.zip),解压到当前目录，得到data文件夹

下载[10000张图像的测试数据](http://pbv7wun2s.bkt.clouddn.com/mix-test-imgs-10000.tar),解压到当前目录

下载[100张图像的测试数据](http://pbv7wun2s.bkt.clouddn.com/mix_test_imgs_100.tar),解压到当前目录

解压后的测试图像的文件夹中，有一个filelist.txt文件，可以使用 
```Shell
mv [images_diretory]/filelist.txt data/
```
命令将图片列表文件取出


### 3、运行程序
```Shell
./main X      其中X=[1,2,3]，表示生成第几个模型的engin(不执行推理)；X=0,执行推理
```

## Int8校验
### 1、下载ImageNet Val校验数据及原始模型

下载[校验数据](http://pbv7wun2s.bkt.clouddn.com/imagenet_val_imgs_txt.zip),解压到当前目录

下载[原始模型](http://pbv7wun2s.bkt.clouddn.com/resnet50_caffemodel.zip),解压到当前目录 

### 2、生成格式化数据集batches
```Shell
cd .. & python  generateBatchFile.py 
```
### 3、校验并生成Int8的engin,并与FP32进行速度对比 
```Shell
cd build & ./CalibrationInt8 Resnet50
```




