# RefineDet
____________

## 使用方式

### 0. 按照Troan-Shadow的README安装相关依赖

### 1. 在Vsepa目录下执行以下命令编译项目

```Shell
mkdir build
cd build
cmake ..
make
```

### 2. 下载[TensorRT版的refinedet模型及测试文件](http://pa5f6jv84.bkt.clouddn.com/redinedet_files.zip) ，同样解压到刚刚建立的build目录下


### 3. 执行./main X 执行网络测试，X=1表示生成bin文件（第一次执行必须生成），不输出X表示不生成


### 结果(images/cat.jpg)

| image_id | class_id | confidence | x_min | y_min | x_max | y_max |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 0 | 16 | 0.993658 | 0.349489 | 0.0514449 | 0.71047 | 0.968079 |
| 0 | 59 | 0.0301644 | 0.683359 | 0.132404 | 0.993784 | 0.783629 |
| 0 | 59 | 0.0254785 | 0.347089 | 0.0464823 | 0.748795 | 0.889772 | 
| 0 | 14 | 0.0253663 | 0.336295 | 0 | 0.736112 | 0.963971 |
| 0 | 59 | 0.0234916 | 0.00987725 | 0.275493 | 0.265681 | 0.883872 |
| 0 | 15 | 0.0187655 | 0.336679 | 0.0436744 | 0.714307 | 0.928742 | 
| 0 | 1  | 0.0184293 | 0.328268 | 0.0654826 | 0.746033 | 0.976873 | 
| 0 | 17 | 0.0145825 | 0.328268 | 0.0654826 | 0.746033 | 0.976873 | 
| 0 | 59 | 0.0137886 | 0.00381823 | 0.0250945 | 0.355265 | 0.766798 |
| 0 | 57 | 0.0132664 | 0.350192 | 0.0203547 | 0.718042 | 0.992188 | 
| 0 | 16 | 0.0132644 | 0.535769 | 0.0447022 | 0.937712 | 0.942751 | 
| 0 | 61 | 0.0114315 | 0.0803385 | 0.750168 | 0.904409 | 1 |


### 相关说明
  1. 插件实现参考[refinedet](https://github.com/sfzhang15/RefineDet)中detectionOutput相关部分
  2. 插件中对loc的编码即ApplyArmLoc插件只实现了参数与detectionOutput中参数一致的情况，即当detectionOutput中参数发生变化，插件可能需要修改。
  3. 图片尺寸，batchSize通过engine直接获得，不需要初始化时赋值
  4. 将dim, buffer中的index, 与数据指针封装为struct
  5. 程序根据engine.getNbBindings()自动计算输出的blob个数，因为默认input blob个数为1


