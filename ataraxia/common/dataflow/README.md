# AVA上的数据集转化工具

## Preparations

    pip install qiniu 


## json_tool_cls.py
当前功能：
  将用于图像分类算法训练的图片数据批量上传至指定bucket，并生成ava json
适用的对象是当前已经挂载在PVC上的，按照MXNet或Caffe格式生成好的用于训练的数据。目前有2种形式的图像训练数据: Caffe的分类数据，MXNet的分类数据(以后有新的可以添加)。
* Caffe 的分类数据格式：
  <file> <cls(int)>
* MXNET 的分类数据格式:
  <index(int)> <label(float)> <file>


### Usage: 将src参数制定的图片list上传至dest参数指定的bucket，并生成json list

	python json_tool_cls.py \
	--root /Users/linyining/Documents/code/atlab/ataraxia/common/dataflow \
	--src cf.list --dest lyn-test --type cf --synset syn_test.list \
	--datasetlabel cls-test --jsonlist ava1.json \
	--ak xxx \
	--sk xxx

### 参数说明：
* root 图片list的根目录，图片list中的图片路径可以是绝对路径也可以是相对路径，如果是绝对路径，这个参数可以不写，默认为''，如果是相对路径，则该参数填相对路径的根目录
* src 图片和标签的list，目前支持caffe的分类格式和mxnet的分类格式两种
* dest 上传的目标bucket
* type 图片和标签的list的格式，目前支持caffe的分类格式和mxnet的分类格式两种，caffe格式对应关键字为"cf"，mxnet格式对应关键字为"mx"
* synset label对应的标签数组文件，示例为：

	pulp
	sexy
	normal

* op 使用的工具的操作，0: 上传图片并生成jsonlist上传，1: 仅生成jsonlist(用于测试)
* datasetlabel ava json中label中的 "name"关键字
* jsonlist 生成的json文件的名字，生成后会自动上传至图片的bucket
* ak, sk: 存储的ak sk

## json_tool_det.py
  将用于图像检测算法训练的图片数据批量上传至指定bucket，并生成ava json
适用的对象是当前已经挂载在PVC上的，按照VOC/ImageNet格式或COCO格式的检测数据(支持jpg和JPEG两种后缀格式)。

### VOC 的检测数据格式：

```
<vocdirname>
├── Annotations
│   ├── 000000.xml
│   ├── 000001.xml
│   ├── 000002.xml
│   └── ...
├── ImageSets
│   ├── Main
│   │   ├── test.txt
│   │   ├── train.txt
│   │   ├── trainval.txt
│   │   └── val.txt
├── JPEGImages
│   ├── 000000.jpg
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
```

### ImageNet 的检测数据格式：

```
<imagenetdirname>
├── Annotations
│   ├── train
│   │   ├── train_000000.xml
│   │   ├── train_000001.xml
│   │   ├── train_000002.xml
│   │   └── ...
│   ├── val
│   │   ├── val_000000.xml
│   │   ├── val_000001.xml
│   │   ├── val_000002.xml
│   │   └── ...
├── ImageSets
│   ├── Main
│   │   ├── test.txt
│   │   ├── train.txt
│   │   ├── trainval.txt
│   │   └── val.txt
├── JPEGImages
│   ├── train
│   │   ├── train_000000.jpg
│   │   ├── train_000001.jpg
│   │   ├── train_000002.jpg
│   │   └── ...
│   ├── test
│   │   ├── test_000000.jpg
│   │   ├── test_000001.jpg
│   │   ├── test_000002.jpg
│   │   └── ...
│   ├── val
│   │   ├── val_000000.jpg
│   │   ├── val_000001.jpg
│   │   ├── val_000002.jpg
│   │   └── ...
```

### COCO 的检测数据格式：

```
<cocodirname>
├── annotations
│   ├── instances_val2017.json
│   └── instances_train2017.json
├── val2017
│   ├── val_000000.jpg
│   ├── val_000001.jpg
│   ├── val_000002.jpg
│   └── ...
├── train2017
│   ├── train_000000.jpg
│   ├── train_000001.jpg
│   ├── train_000002.jpg
│   └── ...
```

### Usage: 将src参数制定的图片list上传至dest参数指定的bucket，并生成json list


    python json_tool_det.py \
    --root /Users/linyining/Documents/code/VOCdevkit/VOC2007 \
    --src cf.list --dest lyn-test --type cf --synset syn_test.list --dataformatflag voc \
    --datasetlabel det-test --jsonlist ava1.json \
    --ak xxx \
    --sk xxx

### 参数说明：
* root 检测数据的根目录，以VOC格式为例，根目录包含 Annotations, JPEGImages 等子目录
* src 数据源
  - VOC/ImageNet格式的数据支持两种格式，一种是VOC格式的根目录(填与"root" 参数相同即可)，另一种是如"VOCdevkit/VOC2007/ImageSets/Layout/train.txt"的list, 每一行对应图片和xml文件的index；
  - COCO格式的数据支持一种格式，即“COCO/annotations/instances_train2017.json”的jsonflie
* dataformatflag 数据格式标识符， 默认按VOC格式，如需处理COCO格式的数据，添加"--dataformatflag coco"，另imagenet格式数据仍选择voc（兼容处理2种格式）
* dest 上传的目标bucket
* skip 需要忽略的类别，目前仅支持单类的string类型，暂不支持数组
* op 使用的工具的操作，0: 上传图片并生成jsonlist上传，1: 仅生成jsonlist(用于测试)
* datasetlabel ava json中label中的 "name"关键字
* jsonlist 生成的json文件的名字，生成后会自动上传至图片的bucket
* ak, sk: 存储的ak sk

### AVA数据集构建：
参见 [ava_dataset_building](ava_dataset_building/README.md)
