# Atlab SDK TEAM Inference 3d人脸landmark网络

### 本项目主要为人脸3Dlandmark网络的TensorRT加速版本

#### 1、编译文件

mkdir build

cd build

cmake ..

make

#### 2、准备测试数据

下载[网络模型](http://pa5n61i2f.bkt.clouddn.com/face.pb.uff.zip), 解压到build目录下，得到face.pb.uff模型文件


下载[测试数据](http://pa5n61i2f.bkt.clouddn.com/test_images.zip), 解压到build目录下，得到image测试图像数据文件夹

#### 3、运行程序

在build目录下执行

./main  其中X=1，表示需要生成engin；X=0,表示不生成engin，直接加载已有engin

landmark结果保存在build目录下landmark.json，测试图片landmark可视化结果保存在build目录下的plot_kpt文件夹内

### 测试图片结果示例

 ![result1](result_img/result_img.jpg)
 
 注：主函数中的人脸检测json结果为七牛线上人脸检测API结果，按照图片命名顺序（按c++sort函数排序规则进行排序，即按命名中每一字符的升序顺序，如本测试集中的命名顺序为1，10，2，3，4，5，6，7，8，9, ）依次赋值。最终的landmark结果也将按此顺序进行存储。
 
