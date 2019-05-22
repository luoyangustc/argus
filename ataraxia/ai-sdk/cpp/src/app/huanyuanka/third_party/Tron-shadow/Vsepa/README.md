# Vsepa网络TensorRT版本（测试版）

该项目将Vsepa网络进行TensoRT版本实现，暂时主要是为了测试Slice插件层正确性以及Vsepa网络实现过程中的基本问题，为以后完整模型出来后进一步开发做基础

## 使用方法：

### 0. 按照Troan-Shadow的README安装相关依赖

### 1. 在Vsepa目录下执行以下命令编译项目

```Shell
mkdir build
cd build
cmake ..
make
```

### 2. 下载[TensorRT版的vsepa模型及测试文件](http://pa5f6jv84.bkt.clouddn.com/vsepa_files.zip) ，同样解压到刚刚建立的build目录下

### 3. 执行./main X 执行网络测试，X=1表示生成bin文件（第一次执行必须生成），不输出X表示不生成


### 说明：

由于模型未提供测试图像以及前后处理过程，因此只能对比最后一层prob-attr的结果或进行压力测试，输入数据为随机生成或手动设置

TensorRT版本的Vsepa相比Caffe原版删除了部分reshape层，将Scale层替换为Eltwise层，将三输入的Eltwise层进行了拆分


