<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1. RTSP容器配置](#1-rtsp%E5%AE%B9%E5%99%A8%E9%85%8D%E7%BD%AE)
  - [1.1.定制和创建RTSP容器镜像](#11%E5%AE%9A%E5%88%B6%E5%92%8C%E5%88%9B%E5%BB%BArtsp%E5%AE%B9%E5%99%A8%E9%95%9C%E5%83%8F)
  - [1.2.运行RTSP容器](#12%E8%BF%90%E8%A1%8Crtsp%E5%AE%B9%E5%99%A8)
- [2. RTSP容器服务使用](#2-rtsp%E5%AE%B9%E5%99%A8%E6%9C%8D%E5%8A%A1%E4%BD%BF%E7%94%A8)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1. RTSP容器配置

## 1.1.定制和创建RTSP容器镜像

```shell
# 可选
# 修改settings.coffee, 默认配置为

module.exports =
    serverPort: 80   # 该端口同时支持 rtsp/rtmp
    rtmpServerPort: 1935  # rtmp还可以有额外的一个端口
    serverName: 'qiniu-rtsp-rtmp-server'

更多配置请参考:
https://github.com/iizukanao/node-rtsp-rtmp-server/blob/master/config.coffee


# 创建RTSP容器镜像
cd docker/rtsp-server

docker build -t rtsp-server .
```

## 1.2.运行RTSP容器

```shell
# 端口和上面保持一致
docker run --name rtsp -d -t -p 80:80 -p 1935:1935 rtsp-server
```

# 2. RTSP容器服务使用 
> 由于此库当前只处于维护状态, 对于解析本地的视频流有bug, 目前已知的mp4有些type没有进行处理, https://github.com/iizukanao/node-rtsp-rtmp-server/blob/master/mp4.coffee#L852, 此处会报错.  **此处不考虑本地视频流的服务**

```shell
# accept stream
ffmpeg -re -i input.mp4 -c:v copy -c:a copy -f flv rtmp://localhost/live/STREAM_NAME
ffmpeg -re -i input.mp4 -c:v copy -c:a copy -f flv rtmp://localhost:1935/live/STREAM_NAME
ffmpeg -re -i input.mp4 -c:v copy -c:a copy -f rtsp rtsp://localhost:80/live/STREAM_NAME

# serving stream
ffmpeg -re -i rtmp://localhost/live/STREAM_NAME -c:v copy -c:a copy -f flv output.mp4
ffmpeg -re -i rtmp://localhost:1935/live/STREAM_NAME -c:v copy -c:a copy -f flv output.mp4
ffmpeg -re -rtsp_transport tcp -i rtsp://localhost:80/live/STREAM_NAME  -c:v copy -c:a copy -f flv output.mp4
```
