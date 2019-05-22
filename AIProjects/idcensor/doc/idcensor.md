<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [基本参数](#%E5%9F%BA%E6%9C%AC%E5%8F%82%E6%95%B0)
  - [输入图片格式](#%E8%BE%93%E5%85%A5%E5%9B%BE%E7%89%87%E6%A0%BC%E5%BC%8F)
- [API](#api)
  - [/v1/id/censor](#v1idcensor)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 基本参数

## 输入图片格式
* 支持`JPG`、`PNG`、`BMP`

* 图片资源表示方式（URI）。通过统一方式定位、获取资源（图片、二进制数据等）
    * HTTP， 网络资源，形如：http://host/path、https://host/path
    * Data，Data URI Scheme形态的二进制文件，形如：data:application/octet-stream;base64,xxx。ps: 当前只支持前缀为data:application/octet-stream;base64,的数据

# API

* __HOST__: http://argus.atlab.ai
* Authorization: qiniu/mac

| PATH | Note | Input | Response Type |
| :--- | :--- | :--- | :---: |
| [`/v1/custom/ai4idcensor/v1/id/censor`](#/v1/custom/ai4idcensor/v1/id/censor) | 企业内部审核人证比对 | image URI | Json |

## /v1/id/censor

> 企业内部审核人证比对

> 通过个人手持身份证正面照和公安内部(hang hui)数据库来判断真人与公安数据库中的身份证头像是否一致<br>
> 由于公安数据库查询接口为异步，需要首先调用check接口提交查询任务将数据调到外部数据库，然后调用detail接口获取信息；<br>
> 若身份证信息已经被提取到了外部数据库则，detail会返回正确信息，否则返回未查到；一个新身份证信息从被check到被提取到外部使detail能拿到平均需要15s<br>

Request

```
POST /v1/custom/ai4idcensor/v1/id/censor  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
	    "uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg" 
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|


Response

```
200 ok

{
    "code": 0,
    "message": "",
    "result": {
        "id"  : "421121199345682432",
        "same": false,
        "similarity": 0.070454694
    }
}
```

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|id|string|ocr识别到的身份证号码|
|same|bool|是否为同一个人，由/v1/face/sim接口返回判别结果|
|similarity|float32|相似度|
