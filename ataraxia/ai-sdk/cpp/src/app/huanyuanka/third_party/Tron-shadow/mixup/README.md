# MIXUP

本项目主要为MIXUP的TensorRT加速版本

## 推理
### 1、编译文件
```Shell
mkdir build
cd build
cmake ..
make
```


### 2、准备测试数据（默认执行路径在 build 下，所以请将文件解压至 build 目录下）

下载[网络模型](http://pbv7wun2s.bkt.clouddn.com/mixup-caffe-models-1121.zip),解压到当前目录

下载[10000张图像的测试数据](http://pbv7wun2s.bkt.clouddn.com/mix-test-imgs-10000.tar),解压到当前目录

下载[100张图像的测试数据](http://pbv7wun2s.bkt.clouddn.com/mix_test_imgs_100.tar),解压到当前目录

解压后的测试图像的文件夹中，有一个filelist.txt文件，可以使用 
```Shell
cp [images_diretory]/filelist.txt data/
```

### 3、运行程序
```Shell
./main X      其中X=[1,2,3]，表示生成第几个模型的engin(不执行推理)；X=0,执行推理
```

### 4、接口说明


```C
ShadowStatus MixupNet::predict(const std::vector<cv::Mat> &imgs, 
                     std::vector<std::string> &outputlayer, 
                     std::vector<std::vector<float> > &results,
                     int enginIndex = 0)
```

1. engineIndex: 
    - 取值范围[0, 1 ,2], 对应init输入时的模型个数与顺序
2. outputlayer:
    - coarse => 取值范围: ['prob']
    - fine => 取值范围: ['prob']
    - det => 取值范围: ['detection_out']
3. reults: 
    - predict前，results中的各个vectord的size需大于所需的size
    - coarse => imgs.size() * 7
    - fine => imgs.size() * 48
    - det => imgs.size() * 500 * 7
    - 在det的检测结果中，每张图片会有500组数，每组7个值。每组中的第2个值表示label，第3个值是conf，第4到7个是locations。label == -1表示背景。500组数会将label != -1的结果放在前面。






