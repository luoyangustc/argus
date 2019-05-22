# Atlab SDK TEAM Inference LIB CPN

本项目主要为cpn网络tensorRT加速版


### 1、编译文件
```Shell
mkdir build
cd build
cmake ..
make
```


### 2、准备测试数据

下载[网络模型](http://pa5n61i2f.bkt.clouddn.com/cpn_freeze.pb.uff.zip)，并解压到build目录


下载[测试数据](http://pa5n61i2f.bkt.clouddn.com/images.zip),并解压至build目录

### 3、运行程序

在build目录下执行
```Shell
./sampleCPN X      其中X=1，表示需要生成engin；X=0,表示不生成engin，直接加载已有engin
```






