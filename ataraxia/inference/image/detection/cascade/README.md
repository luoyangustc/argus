# terror cascade classification
## 模型用途：
该模型用于 暴恐检测 检出物体的 细化分类；目前细化是指：将检测出的刀和枪 细化为 真刀 真枪。
## 使用流程：
将检测模型检测出的刀/枪类别的图像区域bbox，作为分类模型的输入，产生分类结果:
### 分类类别
    0,knives_true 真刀
    1,knives_false 假刀
    2,knives_kitchen 厨房刀
    3,guns_true 真枪
    4,guns_anime 动漫枪
    5,guns_tools 玩具枪
### 算法模型描述
基于官方的caffe版本框架，se-resnet-50模型，准确率为：90.8%
### 级联分类模型 对 刀/枪 分类推理逻辑描述：
对于级联分类模型的输入图片（bbox）：如果其中一个类别的分类score 大于 0.95 那么就将这个类别作为输出结果；否则将 真刀或者真枪 类别作为输出结果。

### 模型文件
http://p3cp748td.bkt.clouddn.com/terror-cascade-cls-201801301209.tar.gz

### 测试数据url
http://p37emf41e.bkt.clouddn.com/terror-cascade-testData-urllist.txt

### 最终的暴恐检测级联分类模型的输出结果是：
对于一张图片，输入到检测模型处理，对于检测到的刀/枪 类别，将刀/枪区域 bbox 取出；输入到 级联分类模型中进行刀/枪的详细类别分类。
最后的检测加级联分类模型的输出结果是：检测出的类别名称和得分（如果是刀/枪则是分类模型的输出类别和得分）
