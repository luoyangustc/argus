## ResNext
### symbol 文件
[symbol_resnext](https://github.com/likelyzhao/mxnet/blob/dev-faster-rcnn/example/rcnn/rcnn/symbol/symbol_resnext.py)
### 使用须知
1. 需要下载 [resnext-101 模型文件](http://data.dmlc.ml/mxnet/models/imagenet/resnext/101-layers/resnext-101-0000.params)
2. 需要修改 [`rcnn/symbol/__init__.py`](https://github.com/likelyzhao/mxnet/blob/dev-faster-rcnn/example/rcnn/rcnn/symbol/__init__.py#L6)
3. 需要修改 [`rcnn/config.py`](https://github.com/likelyzhao/mxnet/blob/dev-faster-rcnn/example/rcnn/rcnn/config.py#L172-L180)
4. 训练或者测试时候 `--network resnext`