# 人脸特征提取

## 下载测试图片和范例模型

```
sh data/get_data.sh
sh models/get_models.sh
```

## 编译子工程Shadow及推理工程Tron
1. 运行一键编译脚本

```
sh scripts/build_shell.sh
```

## 运行
### 用法说明:
```
-----------------------------------------------------
USAGE:
./test_tron <image-path> <face-pts-string> --mtcnn-model <mtcnn-tronmodel-path> --feature-model <feature-model-path>
<image-path>: path to image file
<face-pts-string>: a string in the format as follows:
    '[[20,30],[100,30],[100,110],[20,110]]'
<mtcnn-tronmodel-path>: [optional] path to mtcnn tronmodel
<feature-tronmodel-path>: [optional] path to feature tronmodel
-----------------------------------------------------
```
### 示例:
1.使用默认模型路径：
```
./build/tron/test_tron data/001.jpg [[1499,429],[1576,429],[1576,522],[1499,522]]
```

_（默认路径参数：--mtcnn-model models/mtcnn_merged.tronmodel --feature-model models/resnet34_v1.1_merged.tronmodel）_

2.手动指定模型路径：
```
./build/tron/test_tron data/001.jpg [[1499,429],[1576,429],[1576,522],[1499,522]] --mtcnn-model models/mtcnn_merged.tronmodel --feature-model models/resnet34_v1.1_merged.tronmodel
```

[模型文件](http://p6yobdq7s.bkt.clouddn.com/faceX_feature_tronmodels_20180417.tar)

[测试图像](http://p6yobdq7s.bkt.clouddn.com/images_face_rects_json.tar)

## 返回结果格式范例
1. 通用人脸特征

