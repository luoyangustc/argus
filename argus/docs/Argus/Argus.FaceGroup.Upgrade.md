<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [API](#api)
  - [/face/group/feature](#facegroupfeature)
  - [/face/group/id/feature](#facegroupidfeature)
  - [/face/group/feature/upgrade](#facegroupfeatureupgrade)
  - [/face/group/id/feature/upgrade](#facegroupidfeatureupgrade)
  - [/face/group/id/feature/upgrade/<version>](#facegroupidfeatureupgradeversion)
  - [/face/group/id/feature/rerun](#facegroupidfeaturererun)
  - [/face/group/id/feature/check](#facegroupidfeaturecheck)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# API

| PATH | Note | Input | Response Type |
| :--- | :--- | :--- | :---: |
| [`/v1/face/group/feature`](#/face/group/feature) | 显示所有人脸库的特征版本 | GET | Json |
| [`/v1/face/group/<id>/feature`](#/face/group/<id>/feature) | 显示人脸库的特征版本 | GET | Json |
| [`/v1/face/group/feature/upgrade`](#/face/group/feature/upgrade) | 显示特定状态的所有特征升级 | GET | Json |
| [`/v1/face/group/<id>/feature/upgrade`](#/face/group/<id>/feature/upgrade) | 显示人脸库的所有特征升级 | GET | Json |
| [`/v1/face/group/<id>/feature/upgrade/<version>`](#/face/group/<id>/feature/upgrade/<version>) | 人脸库特征升级 | POST | Json |
| [`/v1/face/group/<id>/feature/rerun`](#/face/group/<id>/feature/rerun) | 人脸库特征重跑（类似升级，除了版本不变） | POST | Json |
| [`/v1/face/group/<id>/feature/check`](#/face/group/<id>/feature/rerun) | 人脸库特征检查，确认备份图片是否存在 | POST | Json |

## /face/group/feature

> 显示所有人脸库的特征版本

Request

```
GET /v1/face/group/feature  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 OK
Content-Type: application/json

{
    "code": 0,
    "message": "",
    "result": [
        {
            "id": "test",
            "feature_version": "v1"
        },
        ...
    ]
}
```


## /face/group/id/feature

> 显示人脸库的特征版本

Request

```
GET /v1/face/group/<id>/feature  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 OK
Content-Type: application/json

{
    "code": 0,
    "message": "",
    "result": {
        "id": "test",
        "feature_version": "v1"
    }
}

```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|


## /face/group/feature/upgrade

> 显示特定状态的所有特征升级

Request

```
GET /v1/face/group/feature/upgrade?status=<string>  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 OK
Content-Type: application/json

{
    "code":0,
    "message":"",
    "result":[
        {
            "id":"test",
            "from":"v1",
            "to":"v2",
            "status":"FINISHED",
            "created_at":"2018-05-04T11:49:40.171197155+08:00",
            "updated_at":"2018-05-04T11:49:40.171197265+08:00"
        },
        ...
    ]
}
```


## /face/group/id/feature/upgrade

> 显示人脸库的所有特征升级

Request

```
GET /v1/face/group/<id>/feature/upgrade  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 OK
Content-Type: application/json

{
    "code":0,
    "message":"",
    "result":[
        {
            "id":"test",
            "from":"v2",
            "to":"v3",
            "status":"UPGRADING",
            "created_at":"2018-05-04T12:49:40.171197155+08:00",
            "updated_at":"2018-05-04T12:49:40.171197265+08:00"
        },
        {
            "id":"test",
            "from":"v1",
            "to":"v2",
            "status":"FINISHED",
            "created_at":"2018-05-04T11:49:40.171197155+08:00",
            "updated_at":"2018-05-04T11:49:40.171197265+08:00"
        },
        ...
    ]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|


## /face/group/id/feature/upgrade/<version>

> 人脸库特征升级

Request

```
POST /v1/face/group/<id>/feature/upgrade/<version>  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 OK
Content-Type: application/json

{
    "code":0,
    "message":"Upgrade on the way",
}
```
返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库唯一标识|
|version|string|特征升级的目标版本|


## /face/group/id/feature/rerun

> 人脸库特征重跑

Request

```
POST /v1/face/group/<id>/feature/rurun  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 OK
Content-Type: application/json

{
    "code":0,
    "message":"Rerun on the way",
}
```
返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库唯一标识|


## /face/group/id/feature/check

> 人脸库特征检查

Request

```
POST /v1/face/group/<id>/feature/check  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 OK
Content-Type: application/json

{
    "code": 0,
    "message": "",
    "result": {
        "id": "test",
        "feature_version": "v1",
        "available": 993,
        "unavailable": 7
    }
}
```
返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库唯一标识|