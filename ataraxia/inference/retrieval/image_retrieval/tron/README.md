# 相似图像搜索功能中图片特征的提取

这个工程采用 vgg19 或 vgg16 网络的fc6层特征作为图片的描述子，后续会通过比较描述子的欧氏距离来判别图片是否相似。<br>
这个工程专门用tron系统和转换过的tron模型，提取fc6层特征。<br>
得到的fc6层特征是4096维float类型，在protobuf里的body里面存在，类型是byte[]<br>

tron模型用下面的命令获得 : <br> 
curl http://iovip.qbox.me/vgg19_merged.tronmodel_online -H 'Host:ovhipn8pb.bkt.clouddn.com' -o vgg19_merged.tronmodel <br>

## 使用方式

### mxnet／caffe模型转换为tron模型
在converter目录下config／config_custom.py中定义网络的一些信息和路径，运行convert.py会进行转换。 <br>



### 编译tron系统
运行./scripts/build_shell.sh,会进行编译，但是注意，先安装一个c++版本的protobuf，同样有protobuf3 和 protobuf2.6.1，我安装protobuf3以后编译cmake工程会提示所用protobuf3版本太高，所以我们实际需要安装的是
protobuf2.6.1。 具体做法参照 https://gist.github.com/samklr/0b8a0620f82005e7f556。 之后应该能顺利编译，会有些warning，但是不伤大雅。<br>
有问题看 https://cf.qiniu.io/pages/viewpage.action?pageId=64919793 <br>

### 准备数据
在目录下创建目录，然后把你想要测试的图片放进去。

### 回归测试
接下来我们就可以看转化的tron模型能不能tron系统shadow引擎跑起来了，实际上就是运行 ./build/tron/test_tron 。但是别急，需要先设好路径等参数，打开你inference下的tron/tron/examples/test_tron.cpp，这个就是main函数。<br>
这里只能看代码了，但是主要流程就是 net 启动，图片进入， predict，以及 netInference 里面的输出，这里没有输出，可以把／tron/serving/infer_algorithm.cpp中的 // res->set_result(result_json_str) 这句话的uncomment去掉;<br>

### 回归测试结果
https://cf.qiniu.io/pages/viewpage.action?pageId=66159865

## 工程的作用
这个模块主要是为了生成.so文件，提供给serving部门，serving会利用.so和你的tron模型，来获得结果，最终用protobuf协议传输。<br>

protobuf的好处是把数据二进制化，会很小。protobuf定义的message结构在.proto文件中定义，然后编译下会生成 pb.c / .cc/ .h 文件（生成代码），里面有你定义的数据他如何被传递到protobuf的函数接口。<br>

例如这个例子新定义了一个结构，在 proto/.proto里面 optional bytes body =6; 这里 bytes 是protobuf的传递类型，编译后会在tron/proto/inference.pb.h中产生<br>

inline void set_body(const ::std::string& value);<br>
inline void set_body(const char* value);<br>
inline void set_body(const void* value, size_t size);<br>

这三个传递的接口，接下来你只需要在tron/serving/infer_algorithm.cpp 把你的数据传递进去就好，例如我这边是 vector<float> 类型，<br>

const void* protobuf_bytes = reinterpret_cast<const void*>(&classification_output.scores[0]);<br>
int size_classification = (classification_output.scores.size())*4;<br>
res->set_body(protobuf_bytes,size_classification);<br>
