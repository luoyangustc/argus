# Ataraxia
用来放算法所有公有云上线的推理代码，不同算法大类间公共的基础代码，
## 目录结构
基础目录如下，如有增加请修改本文档

- ava/  放 `ava serving` 相关的内容
- doc/  放各种类型的文档
- common/ 公共代码
	- tron/ 放 `TRON` 的代码库
- dockerfiles/ 放算法公共镜像的Dockerfiles
	- caffe/
	- mxnet/
	- tensorflow/
	- pytorch/
	- caffe2/
- inference
	- ocr/ `ocr` 项目推理代码 
		- python/ `python` 版本推理
		- tron/	`tron` 	版本推理
	- image/ 图片相关推理代码
		- classification/ 分类代码
		- detection/ 检测代码
	- video/ 视频技术相关的代码
	- retrieve/ 

## 结构要求
1. 算法推理代码放在`inference`文件夹下
2. `inference`文件夹下不同的算法放在不同的文件夹下
3. 每种算法下面有`python` 和 `tron` 用来放不同语言的推理代码

