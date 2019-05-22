# 架构设计

处理流程:

1. 输入图片
2. 图像分类：区分图片为长微博／微信聊天记录／其他文本文字／其他
3. 文字检测：检测图片中的文字区域
4. (not available yet) 筛选出面积比重符合要求的（area_ratio_thresh）
5. 文字识别：识别各文字区域的文字内容，并进行拼接等操作
6. 后处理：文字识别内容纠正，词频统计

# 环境配置

1. dockerfile: [Dockerfile](Dockerfile)
   从dockerfile build一个镜像，然后新建一个容器。也可以直接拉取我build好的镜像：reg-xs.qiniu.io/atlab/ctpn:20180118-torch0.3.0

2. 下载当前repo

   ```
   cd ataraxia/ocr/weixin-weibo/text-detection
   make
   ```

3. 下载图像分类、文字检测和文字识别的模型，分别在text-classification/models、text-detection/models和text-recognition/models目录下

   图像分类模型：<http://p1d01vydp.bkt.clouddn.com/text-classification-v0.2-t3.caffemodel>

   文字检测模型：<http://osaoorkjs.bkt.clouddn.com/ctpn_trained_model.caffemodel>

   文字识别模型：<http://osaoorkjs.bkt.clouddn.com/netCRNN_v4_0_110000.pth>

4. 配置模型信息：在text-classification/cfg.py、text-detection/cfg.py和text-recognition/cfg.py中配置模型名称等信息



## API定义

1. 图像分类：

   ```
   cd text-classification
   python demo_classification.py --image YOUR_IMAGE_PATH
   ```
返回：
   ```python
   {
      img: <string>,    #输入图片路径
      img_type: <string>  #blog/wechat/other-text/others
   }
   ```

2. 文字检测：

   ```
   cd text-detection
   python demo_detect.py --image YOUR_IMAGE_PATH
   ```
返回：
   ```python
   {
      img: <string>,    #输入图片路径
      bboxes: <array>, (e.g.: [[x0,y0,x1,y1],[...],...] )  #输出检测到的文字区域
      area_ratio: <float>,   #输出文字区域在全图中所占的面积比例
      img_type: <string>
   }
   ```

3. 文字识别

   ```
   cd text-recognition
   python demo_recog.py --image YOUT_IMAGE_PATH
   ```
返回：
   ```python
   {
       img: <string>,
       text: <string>
   }
   ```

4. 后处理,词频统计
```
  cd text-frequency
  python  word_static.py --image YOUT_IMAGE_PATH 
```
返回：
   ```python
   {
       [
       ["word",word_frequency]
        ...
       ]
   }
   ```
