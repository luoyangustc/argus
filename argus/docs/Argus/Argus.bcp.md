<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [人脸聚类](#%E4%BA%BA%E8%84%B8%E8%81%9A%E7%B1%BB)
  - [API](#api)
  - [image-censor](#image-censor)
  - [bucket-inspect](#bucket-inspect)
  - [bucket/up](#bucketup)
    - [回调规格](#%E5%9B%9E%E8%B0%83%E8%A7%84%E6%A0%BC)
  - [bucket/stock](#bucketstock)
  - [/v1/bucket/rules/set](#v1bucketrulesset)
  - [/v1/bucket/rules/delete](#v1bucketrulesdelete)
  - [/v1/bucket/rules](#v1bucketrules)
  - [POST/v1/bucket/jobs](#postv1bucketjobs)
    - [hookurl的结果格式](#hookurl%E7%9A%84%E7%BB%93%E6%9E%9C%E6%A0%BC%E5%BC%8F)
  - [/v1/bucket/jobs/xx/cancel](#v1bucketjobsxxcancel)
  - [GET/v1/bucket/jobs](#getv1bucketjobs)
  - [/v1/bucket/jobs/xx](#v1bucketjobsxx)
  - [/v1/bucket/last](#v1bucketlast)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# 人脸聚类

## API

| 入口 | PATH | 说明 |
| :--- | :--- | :--- |
| FOP | [`image-censor`](#image-censor) | 图片审核 |
| PFOP | [`bucket-inspect`](#bucket-inspect) | 设置disable能力 |
| PFOP | [`image-censor|bucket-inspect`](#bucketup) | bucket的增量审核 |
| BJOB | [`bucket-censor`](#bucketstock) | bucket的存量审核 |
| BCP | [`POST /v1/bucket/rules/set`](#v1bucketrulesset) | 设置bucket增量审核 |
| BCP | [`POST /v1/bucket/rules/delete`](#v1bucketrulesdelete) | 删除bucket增量审核 |
| BCP | [`GET /v1/bucket/rules`](#v1bucketrules) | 获取bucket增量审核设置 |
| BCP | [`POST /v1/bucket/jobs`](#POSTv1bucketjobs) | 触发bucket存量审核 |
| BCP | [`POST /v1/bucket/jobs/xx/cancel`](#v1bucketjobsxxcancel) | 停止bucket存量审核 |
| BCP | [`GET /v1/bucket/jobs`](#GETv1bucketjobs) | 获取bucket存量审核任务 |
| BCP | [`GET /v1/bucket/jobs/xx`](#v1bucketjobsxx) | 获取指定的bucket存量审核任务 |
| BCP | [`GET /v1/bucket/last`](#v1bucketlast) | 获取指定的bucket审核现状 |

## image-censor

```
FOP ?image-censor/pulp/terror/politician
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `pulp` | 否 | 指定执行剑皇审核 |
| `terror` | 否 | 指定执行暴恐审核 |
| `politician` | 否 | 指定执行涉政审核 |
| | 否 | 上述参数都不选的情况，默认三审全部执行。即`image-censor`等效于`image-censor/pulp/terror/politician` |


## bucket-inspect

```
PFOP ?bucket-inspect/v1/tpulp/<tpulp>/tsexy/<tsexy>/tterror/<tterror>/tpolitician/<tpolitican>
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `v1` | 是 | 版本号，当前为`v1` |
| `tpulp` | 否 | 指定disable的涉黄阈值 |
| `tsexy` | 否 | 指定disable的涉性感阈值 |
| `tterror` | 否 | 指定disable的涉暴阈值 |
| `tpolitician` | 否 | 指定disable的涉政阈值 |

## bucket/up

```
PFOP ?image-censor|bucket-inspect
```

### 回调规格

*Request*

```
POST /xxxx  Http/1.1
Content-Type: application/json

{
    "id": "z0.0AC8145700002A9F0000000AD61B9CD5", 
    "pipeline": "0.default", 
    "code": 0, 
    "desc": "The fop was completed successfully", 
    "reqid": "Wp4AAIxsWr63_i4V", 
    "inputBucket": "censor3", 
    "inputKey": "filter6/test2/pulp-test3.png", 
    "items": [
        {
            "cmd": "image-censor/pulp/terror/politician|bucket-inspect/v1/pulp/0.98/sexy/1/terror/0.60/politician/0.60", 
            "code": 0, 
            "desc": "The fop was completed successfully", 
            "result": {
                "disable": true, 
                "result": {
                    "code": 0, 
                    "message": "", 
                    "result": {
                        "label": 1, 
                        "review": false, 
                        "score": 0.99426526
                        "details": [
                            {
                                "label": 0, 
                                "review": false, 
                                "score": 0.99426526, 
                                "type": "pulp"
                            }, 
                            {
                                "label": 0, 
                                "more": [
                                    {
                                        "boundingBox": {
                                            "pts": [
                                                [
                                                    861, 
                                                    106
                                                ], 
                                                ...
                                            ], 
                                            "score": 0.99951386
                                        }, 
                                        "value": {
                                            "review": false, 
                                            "score": 0.22625886
                                        }
                                    }
                                ], 
                                "review": false, 
                                "score": 0, 
                                "type": "politician"
                            }, 
                            {
                                "label": 0, 
                                "review": false, 
                                "score": 0.8930834, 
                                "type": "terror"
                            }
                        ], 

                    }
                }
            }, 
            "returnOld": 0
        }
    ]
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|items.result.result|||参考三鉴的结果返回文档|
|disable|bool|文件是否被disable|

## bucket/stock

*Request*

```
POST /v1/submit/bucket-censor HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "hookURL": <hookURL>,
    "request": {
        "bucket": <bucket>,
        "prefix": <prefix>,
        "params": {
            "type": ["pulp", "terror", "politician"]
        },
        "save": {
            "bucket": <bucket>,
            "prefix": <prefix>
        }
    }
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `hookURL` | 否 | 回调地址 |
| `request.bucket` | 是 | 需要扫描的bucket |
| `request.prefix` | 否 | 需要扫描的bucket前缀 |
| `request.params` | 否 | 指定的请求图片审核(image-censor)的参数 |
| `request.save` | 否 | 指定保存结果 |
| `request.save.bucket` | 否 | 指定保存结果的bucket |
| `request.save.prefix` | 否 | 指定保存结果的bucket前缀 |

*Response*

```
200 OK
Content-Type: application/json

{
    "job_id": <jobID>
}
```

## /v1/bucket/rules/set

*Request*

```
POST /v1/bucket/rules/set HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "bucket": {
        "bucket": <bucket>,
        "prefix": <prefix>,
        "mimeType": "IMAGE"
    },
    "mode": <mode>,
    "censorType": [],
    "disable": {
        "on": true,
        "threshold": {
            "pulp": {
                "pulp": 0.66,
                "sexy": 0.65
            },
            "terror": 0.55
        }
    },
    "notify": {
        "notifyUrl": <notifyUrl>
    },
    "pfop": {
        "name": <name>,
        "pipeline": <pipeline>
    },
    "cover": true
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 是 | bucket信息 |
| `bucket.bucket` | 是 | bucket名 |
| `bucket.prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `bucket.mimeType` | 是 | `IMAGE`，当前只支持图片 |
| `mode` | 是 | `0x01`：机审 |
| `censorType` | 否 | 审核类型，可选项`pulp`、`terror`、`politician`；不填表示三审 |
| `disable` | 否 | 设置违规disable功能 |
| `disable.on` | 否 | 是否开启违规disable功能 |
| `disable.threshold` | 否 | 设置违规阈值，值的类型为`float32`或者`map[string]float32` |
| `notify` | 否 | 设定结果配置 |
| `notify.notifyUrl` | 否 | 设定结果回调地址 |
| `pfop` | 否 | 设定`pfop`设置 |
| `pfop.name` | 否 | 设定该规则对应`pfop`设置名称 |
| `pfop.pipeline` | 否 | 设定对应`pfop`的队列名称 |
| `cover` | 否 | 是否覆盖之前的设置（`bucket+prefix+mimeType`)，默认为`false` |

*Response*

```
200 OK
Content-Type: application/json

```

## /v1/bucket/rules/delete

*Request*

```
POST /v1/bucket/rules/delete HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "bucket": {
        "bucket": <bucket>,
        "prefix": <prefix>,
        "mimeType": "IMAGE"
    },
    "createTime": <createTime>
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket.bucket` | 是 | bucket名 |
| `bucket.prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `bucket.mimeType` | 是 | `IMAGE`，当前只支持图片 |
| `create_time` | 是 | 规则创建时间 |

*Response*

```
200 OK

```

## /v1/bucket/rules

*Request*

```
GET /v1/bucket/rules HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "bucket": <bucket>,
    "prefix": <prefix>,
    "mimeType": "IMAGE"
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 否 | bucket名 |
| `prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `mimeType` | 是 | `IMAGE`，当前只支持图片 |

*Response*

```
200 OK
Content-Type: application/json

[
    {
        "bucket": {
            "bucket": <bucket>,
            "prefix": <prefix>,
            "mimeType": "IMAGE"
        },
        "mode": <mode>,
        "censorType": [],
        "disable": {},
        "notify": {},
        "pfop": {},
        "createTime": <createTime>,
        "endTime": <endTime>
    },
    ...
]
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 是 | bucket信息 |
| `bucket.bucket` | 是 | bucket名 |
| `bucket.prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `bucket.mimeType` | 是 | `IMAGE`，当前只支持图片 |
| `mode` | 是 | `0x01`：机审 |
| `censorType` | 否 | 审核类型，可选项`pulp`、`terror`、`politician`；不填表示三审 |
| `disable` | 否 | 设置违规disable功能 |
| `notify` | 否 | 设定结果配置 |
| `pfop` | 否 | 设定pfop配置 |
| `createTime` | 是 | 起始时间，单位纳秒 |
| `endTime` | 否 | 结束时间，单位纳秒 |

## POST/v1/bucket/jobs

*Request*

```
POST /v1/bucket/jobs HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "bucket": {
        "bucket": <bucket>,
	    "prefix": <prefix>,
	    "mimeType": "IMAGE"
    },
    "mode": <mode>,
    "censorType": [],
    "disable": {
        "on": true,
        "threshold": {
            "pulp": {
                "pulp": 0.66,
                "sexy": 0.65
            },
            "terror": 0.55
        }
    },
    "notify": {
        "notifyUrl": <notifyUrl>,
        "save": {
            "bucket": <bucket>,
            "prefix": <prefix>
        }
    }
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 是 | bucket信息 |
| `bucket.bucket` | 是 | bucket名 |
| `bucket.prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `bucket.mimeType` | 是 | `IMAGE`，当前只支持图片 |
| `mode` | 是 | `0x01`：机审 |
| `censorType` | 否 | 审核类型，可选项`pulp`、`terror`、`politician`；不填表示三审 |
| `disable` | 否 | 设置违规disable功能 |
| `disable.on` | 否 | 是否开启违规disable功能 |
| `disable.threshold` | 否 | 设置违规阈值，值的类型为`float32`或者`map[string]float32` |
| `notify` | 否 | 设定结果配置 |
| `notify.notifyUrl` | 否 | 设定结果回调地址 |

*Response*

```
200 OK
Content-Type: application/json

{
    "job_id": <jobID>
}
```

### hookurl的结果格式
```
{"keys":["filter7//5afc0cf25f171c000735adee/20180516T185034_20180516T185043__15",...]}
```

| 参数 |  说明 |
| :--- | :--- |
| `keys` | 返回的结果文件路径 |

## /v1/bucket/jobs/xx/cancel

*Request*

```
POST /v1/bucket/jobs/<jobID>/cancel HTTP/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `jobID` | 是 | 认为ID |

*Response*

```
200 OK
Content-Type: application/json

```

## GET/v1/bucket/jobs

*Request*

```
GET /v1/bucket/jobs HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "bucket": <bucket>,
    "prefix": <prefix>,
    "mimeType": "IMAGE"
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 否 | bucket名 |
| `prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `mimeType` | 是 | `IMAGE`，当前只支持图片 |

*Response*

```
200 OK
Content-Type: application/json

[
    {
        "bucket": {
            "bucket": <bucket>,
            "prefix": <prefix>,
            "mimeType": "IMAGE"
        },
        "mode": <mode>,
        "censorType": [],
        "disable": {},
        "notify": {},
        "id: <jobID>,
        "state": <state>,
        "createTime": <createTime>,
        "endTime": <endTime>
        "result": {
            "saveKeys": []
        }
    },
    ...
]
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 是 | bucket信息 |
| `bucket.bucket` | 是 | bucket名 |
| `bucket.prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `bucket.mimeType` | 是 | `IMAGE`，当前只支持图片 |
| `mode` | 是 | `0x01`：机审 |
| `censorType` | 否 | 审核类型，可选项`pulp`、`terror`、`politician`；不填表示三审 |
| `disable` | 否 | 设置违规disable功能 |
| `notify` | 否 | 设定结果配置 |
| `id` | 是 | 任务ID |
| `state` | 是 | 任务状态，`WAITING`、`DOING`、`FINISHED` |
| `createTime` | 是 | 起始时间，单位纳秒 |
| `endTime` | 否 | 结束时间，单位纳秒 |
| `result.saveKeys` | 否 | 结果保存的文件名 |

## /v1/bucket/jobs/xx

*Request*

```
GET /v1/bucket/jobs/<jobID> HTTP/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `jobID` | 是 | 任务ID |

*Response*

```
200 OK
Content-Type: application/json

{
    "bucket": {
        "bucket": <bucket>,
        "prefix": <prefix>,
        "mimeType": "IMAGE"
    },
    "mode": <mode>,
    "censorType": [],
    "disable": {},
    "notify": {},
    "pfop": {},
    "createTime": <createTime>,
    "endTime": <endTime>
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 是 | bucket信息 |
| `bucket.bucket` | 是 | bucket名 |
| `bucket.prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `bucket.mimeType` | 是 | `IMAGE`，当前只支持图片 |
| `mode` | 是 | `0x01`：机审 |
| `censorType` | 否 | 审核类型，可选项`pulp`、`terror`、`politician`；不填表示三审 |
| `disable` | 否 | 设置违规disable功能 |
| `notify` | 否 | 设定结果配置 |
| `id` | 是 | 任务ID |
| `state` | 是 | 任务状态，`WAITING`、`DOING`、`FINISHED` |
| `createTime` | 是 | 起始时间，单位纳秒 |
| `endTime` | 否 | 结束时间，单位纳秒 |
| `result.saveKeys` | 否 | 结果保存的文件名 |

## /v1/bucket/last

*Request*

```
GET /v1/bucket/last HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "bucket": <bucket>,
    "prefix": <prefix>,
    "mimeType": "IMAGE"
}
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket` | 是 | bucket名 |
| `prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `mimeType` | 是 | `IMAGE`，当前只支持图片 |

*Response*

```
200 OK
Content-Type: application/json

[
    {
        "bucket": {
            "bucket": <bucket>,
            "prefix": <prefix>,
            "mimeType": "IMAGE"
        },
        "rule" {
            "mode": <mode>,
            "censorType": [],
            "createTime": <createTime>,
            "endTime": <endTime>
        },
        "job" {
            "mode": <mode>,
            "censorType": [],
            "createTime": <createTime>,
            "endTime": <endTime>
        }
    },
    ...
]
```

| 参数 | 必选 | 说明 |
| :--- | :--- | :--- |
| `bucket.bucket` | 是 | bucket名 |
| `bucket.prefix` | 否 | bucket前缀，限定bucket内的审核范围；不设置表示全部 |
| `bucket.mimeType` | 是 | `IMAGE`，当前只支持图片 |
| `rule` | 否 | 增量审核 |
| `rule.mode` | 是 | `0x01`：机审 |
| `rule.censorType` | 否 | 审核类型，可选项`pulp`、`terror`、`politician`；不填表示三审 |
| `rule.createTime` | 是 | 起始时间，单位纳秒 |
| `rule.endTime` | 否 | 结束时间，单位纳秒 |
| `job` | 否 | 存量审核 |
| `job.mode` | 是 | `0x01`：机审 |
| `job.censorType` | 否 | 审核类型，可选项`pulp`、`terror`、`politician`；不填表示三审 |
| `job.createTime` | 是 | 起始时间，单位纳秒 |
| `job.endTime` | 否 | 结束时间，单位纳秒 |
