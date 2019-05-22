<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Bucket 内容检测](#bucket-%E5%86%85%E5%AE%B9%E6%A3%80%E6%B5%8B)
  - [剑皇](#%E5%89%91%E7%9A%87)
    - [请求规格](#%E8%AF%B7%E6%B1%82%E8%A7%84%E6%A0%BC)
    - [回调规格](#%E5%9B%9E%E8%B0%83%E8%A7%84%E6%A0%BC)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Bucket 内容检测

只能通过Dora异步入口（pfop），配合对应的推理FOP提供审核功能，`cmd`为`bucket-inspect`。

*基本使用方式*

```
POST /pfop HTTP/1.1
Host: api.qiniu.com
Content-Type: application/x-www-form-urlencoded
Authorization: Qbox <AccessKey>:<Sign>

bucket=<bucket>
&key=<key>
&fops=<Eval Cmd>|bucket-inspect/xxxx
&notifyURL=<url>
```

## 剑皇

### 请求规格

```
qpulp|bucket-inspect/v1/pthreshold/0.95/sthreshold/0.99
```

参数说明

|字段|取值|说明|
|:---|:---|:---|
|resource_url||图片资源地址|
|qpulp||剑皇服务的fop cmd|
|v1|string|版本信息|
|pthreshold|float32, 0.0~1.0|剑皇审核参数，当判别为色情图片时，若score大于pthreshold则disable；若未设置pthrehold则色情图片全部disable|
|sthreshold|float32, 0.0~1.0|剑皇审核参数，当判别为性感图片时，若score大于sthreshold则disable；若未设置sthrehold则性感图片不会disable|

### 回调规格

*Request*

```
POST /xxxx  Http/1.1
Content-Type: application/json

{
    "id": <job.id>,
    "pipeline": <pipeline>,
    "code": 0,
    "desc": "The fop was completed successfully",
    "reqid": <reqid>,
    "inputBucket": <bucket>,
    "inputKey": <key>,
    "items": [
        {
            "cmd": "qpulp|bucket-inspect/v1",
            "code": 0,
            "desc": "The fop was completed successfully",
            "result": {
                "disable": true,
                "result": {
                    "code": 0,
                    "message": "",
                    "result": {
                        "label": 0,
                        "review": false,
                        "score": 0
                    }
                }
            },
            "returnOld": 0
        }
    ]
}
```

外层规格同Dora异步回调。

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|items.result.result|同qpulp结果|
|disable|bool|文件是否被disable|
