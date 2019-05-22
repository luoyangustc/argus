<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [简易临时存储系统（Simple Temporary Storage）](#%E7%AE%80%E6%98%93%E4%B8%B4%E6%97%B6%E5%AD%98%E5%82%A8%E7%B3%BB%E7%BB%9Fsimple-temporary-storage)
  - [需求](#%E9%9C%80%E6%B1%82)
  - [设计](#%E8%AE%BE%E8%AE%A1)
  - [使用姿势](#%E4%BD%BF%E7%94%A8%E5%A7%BF%E5%8A%BF)
    - [以代理形式拉取资源](#%E4%BB%A5%E4%BB%A3%E7%90%86%E5%BD%A2%E5%BC%8F%E6%8B%89%E5%8F%96%E8%B5%84%E6%BA%90)
    - [预加载资源](#%E9%A2%84%E5%8A%A0%E8%BD%BD%E8%B5%84%E6%BA%90)
    - [资源传递](#%E8%B5%84%E6%BA%90%E4%BC%A0%E9%80%92)
  - [API](#api)
    - [异步拉取资源](#%E5%BC%82%E6%AD%A5%E6%8B%89%E5%8F%96%E8%B5%84%E6%BA%90)
    - [代理获取资源](#%E4%BB%A3%E7%90%86%E8%8E%B7%E5%8F%96%E8%B5%84%E6%BA%90)
    - [写入资源](#%E5%86%99%E5%85%A5%E8%B5%84%E6%BA%90)
    - [获取资源](#%E8%8E%B7%E5%8F%96%E8%B5%84%E6%BA%90)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# 简易临时存储系统（Simple Temporary Storage）
## 需求

* IO Pipe，读写管道，以支持异步化
* Download Proxy，下载代理，简化资源获取逻辑。

## 设计

* Carp Hash [code](https://github.com/qbox/base/blob/develop/biz/src/qbox.us/dht/carp.go)
* 基于文件系统的小文件存储
	* 效率靠SSD
	* 以时间为第一层目录组织文件
* 内存元数据
* 定时批量清理
	* 每半小时，清理整个目录
* 功能
	* 支持完整文件读写，暂不支持`range get`
	* 支持`fetch`功能，即可异步拉取资源

## 使用姿势

### 以代理形式拉取资源

> 将各种资源拉取进行封装，简化成`HTTP GET`请求

1. 调用[代理获取资源](#代理获取资源)获取资源

### 预加载资源

> 通过后台任务，异步预加载资源

1. 调用[异步拉取资源](#异步拉取资源),`sts`后台预拉取资源
2. 调用[代理获取资源](#代理获取资源)获取资源

### 资源传递

> 一实例向`sts`写入资源，一实例从`sts`获取资源

1. 调用[写入资源](#写入资源)，向`sts`写入资源
2. 调用[获取资源](#获取资源)，从`sts`获取资源


## API

### 异步拉取资源
> `sts`后台自动加载`uri`资源

Request

```
POST /v1/fetch?uri=<uri:base64>&length=<length:int64>&sync=<sync:true|false>
```

Response

```
200 OK
```

* uri: 外部资源地址
* length: 资源大小
* sync: 默认是false，如果是true，则资源拉取成功之后再返回

### 代理获取资源
> 从`sts`获取`uri`对应的已经预加载的资源，如果资源不存在，同步拉取`uri`资源

Request

```
GET /v1/fetch?uri=<uri:base64>&length=<length:int64>
```

Response

```
200 OK
Content-Type: application/octet-stream
Content-Length: <length:int64>

...
```

* uri: 外部资源地址
* length: 资源大小

### 写入资源
> 以`filename`作为标示，上传资源内容

Request

```
POST /v1/file/<filename>
Content-Type: application/octet-stream
Content-Length: <length:int64>

...
```

Response

```
200 OK
```

* filename: 资源名（保证`sts`中唯一）
* length: 资源大小

### 获取资源
> 以`filename`作为标示，下载资源内容

Request

```
GET /v1/file/<filename>
```

Response

```
200 OK
Content-Type: application/octet-stream
Content-Length: <length:int64>

...
```

* filename: 资源名（保证`sts`中唯一）
* length: 资源大小
