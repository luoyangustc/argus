<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [通用审核portal后端服务](#%E9%80%9A%E7%94%A8%E5%AE%A1%E6%A0%B8portal%E5%90%8E%E7%AB%AF%E6%9C%8D%E5%8A%A1)
  - [1. 审核任务管理](#1-%E5%AE%A1%E6%A0%B8%E4%BB%BB%E5%8A%A1%E7%AE%A1%E7%90%86)
    - [1.1 新建审核监控任务](#11-%E6%96%B0%E5%BB%BA%E5%AE%A1%E6%A0%B8%E7%9B%91%E6%8E%A7%E4%BB%BB%E5%8A%A1)
    - [1.2 新建审核文件任务](#12-%E6%96%B0%E5%BB%BA%E5%AE%A1%E6%A0%B8%E6%96%87%E4%BB%B6%E4%BB%BB%E5%8A%A1)
    - [1.3 启动审核任务](#13-%E5%90%AF%E5%8A%A8%E5%AE%A1%E6%A0%B8%E4%BB%BB%E5%8A%A1)
    - [1.4 停止任务](#14-%E5%81%9C%E6%AD%A2%E4%BB%BB%E5%8A%A1)
    - [1.5 更新任务](#15-%E6%9B%B4%E6%96%B0%E4%BB%BB%E5%8A%A1)
    - [1.6 获取任务列表](#16-%E8%8E%B7%E5%8F%96%E4%BB%BB%E5%8A%A1%E5%88%97%E8%A1%A8)
    - [1.7 获取任务历史](#17-%E8%8E%B7%E5%8F%96%E4%BB%BB%E5%8A%A1%E5%8E%86%E5%8F%B2)
  - [2. 审核管理](#2-%E5%AE%A1%E6%A0%B8%E7%AE%A1%E7%90%86)
    - [2.1 获取审核结果](#21-%E8%8E%B7%E5%8F%96%E5%AE%A1%E6%A0%B8%E7%BB%93%E6%9E%9C)
    - [2.2 更改审核结果](#22-%E6%9B%B4%E6%94%B9%E5%AE%A1%E6%A0%B8%E7%BB%93%E6%9E%9C)
    - [2.3 下载审核结果](#23-%E4%B8%8B%E8%BD%BD%E5%AE%A1%E6%A0%B8%E7%BB%93%E6%9E%9C)
    - [2.4 获取视频帧结果](#24-%E8%8E%B7%E5%8F%96%E8%A7%86%E9%A2%91%E5%B8%A7%E7%BB%93%E6%9E%9C)
  - [3. 用户管理](#3-%E7%94%A8%E6%88%B7%E7%AE%A1%E7%90%86)
    - [3.1 新建用户](#31-%E6%96%B0%E5%BB%BA%E7%94%A8%E6%88%B7)
    - [3.2 删除用户](#32-%E5%88%A0%E9%99%A4%E7%94%A8%E6%88%B7)
    - [3.3 更新用户](#33-%E6%9B%B4%E6%96%B0%E7%94%A8%E6%88%B7)
    - [3.4 更新密码](#34-%E6%9B%B4%E6%96%B0%E5%AF%86%E7%A0%81)
    - [3.5 获取用户列表](#35-%E8%8E%B7%E5%8F%96%E7%94%A8%E6%88%B7%E5%88%97%E8%A1%A8)
  - [4. 登录相关](#4-%E7%99%BB%E5%BD%95%E7%9B%B8%E5%85%B3)
    - [4.1 登录](#41-%E7%99%BB%E5%BD%95)
    - [4.2 登出](#42-%E7%99%BB%E5%87%BA)
    - [4.3 获取系统配置](#43-%E8%8E%B7%E5%8F%96%E7%B3%BB%E7%BB%9F%E9%85%8D%E7%BD%AE)
  - [5. 资源管理](#5-%E8%B5%84%E6%BA%90%E7%AE%A1%E7%90%86)
    - [5.1 批量上传资源](#51-%E6%89%B9%E9%87%8F%E4%B8%8A%E4%BC%A0%E8%B5%84%E6%BA%90)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 通用审核portal后端服务

> 除登录相关接口，其他接口均需登录后访问（且必须有对应权限），否则返回http code 401

## 1. 审核任务管理

### 1.1 新建审核监控任务

> 新建审核监控任务，用于采集数据，新建成功后会启动任务

**Request**

```
POST /v1/set/add
Content-Type: application/json

{
    "name": "xxx",
    "type": "monitor_active",
    "scenes": ["pulp","terror","politician"],
    "uri": "http://xxx",
    "monitor_interval": 30,
    "mime_types": ["image","video"],
    "cut_interval_msecs": 1000
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| name | string | 该任务的名称，必选 |
| type | string | 任务的类型，目前支持取值："monitor_active"、"monitor_passive"，必选 |
| scenes | list | 任务的审核场景，取值："pulp","terror","politician"，必选 |
| uri | string | 任务的api路径，任务类型为"monitor_active"时必选 |
| monitor_interval | int | 任务抓取数据的间隔秒数，任务类型为"monitor_active"时必选，最小15秒 |
| mime_types | list | 任务的文件类型，取值："image","video"，必选 |
| cut_interval_msecs | int | 截帧频率，单位：毫秒。取值范围为1000～60000（即1s~60s），mime_types字段包含video时必选 |

**Response**

```
200 OK
Content-Type: application/json
{
   "id" : ""
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :---| :--- | :--- |
| id | string | 创建成功返回的任务id |


### 1.2 新建审核文件任务

> 新建审核文件任务，待审核资源在上传文件中，新建成功后会启动任务
> 文件内容需为每行一条待审核资源url，非url行不会处理

**Request**

```
POST /v1/set/upload
Content-Type: multipart/form-data

form参数：
"name": "xxx"
"scenes": ["pulp","terror","politician"]
"mime_types": ["image","video"]
"cut_interval_msecs": 1000
"file": 上传文件
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| name | string | 该任务的名称，必选 |
| scenes | list | 任务的审核场景，取值："pulp","terror","politician"，必选 |
| mime_types | list | 任务的文件类型，取值："image","video"，必选 |
| cut_interval_msecs | int | 截帧频率，单位：毫秒。取值范围为1000～60000（即1s~60s），mime_types字段包含video时必选  |
| file | string | 表单提交的文件，必选 |

**Response**

```
200 OK
Content-Type: application/json
{
   "id" : ""
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :---| :--- | :--- |
| id | string | 创建成功返回的任务id |


### 1.3 启动审核任务

> 启动审核任务

**Request**

```
POST /v1/set/<id>/start
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 任务id |


**Response**

```
200 OK
```


### 1.4 停止任务

> 停止任务

**Request**

```
POST /v1/set/<id>/stop
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 任务id |


**Response**

```
200 OK
```


### 1.5 更新任务

> 更新任务

**Request**

```
POST /v1/set/<id>/update
Content-Type: application/json

{
    "name": "xxx"
    "scenes": ["pulp","terror","politician"],
    "uri": "http://xxx",
    "monitor_interval": 30,
    "mime_types": ["image","video"],
    "cut_interval_msecs": 1000
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| name | string | 任务的名称，必选 |
| scenes | list | 任务的审核场景，取值："pulp","terror","politician"，必选 |
| uri | string | 任务的api路径，任务类型为"monitor_active"和"monitor_passive"时必选 |
| monitor_interval | int | 任务抓取数据的间隔时间，任务类型为"monitor_active"时必选，最小15秒 |
| mime_types | list | 任务的文件类型，目前支持："image","video"，必选 |
| cut_interval_msecs | int | 截帧频率，单位：毫秒。取值范围为1000～60000（即1s~60s），mime_types字段包含video时必选 |

**Response**

```
200 OK
```

### 1.6 获取任务列表

> 获取任务列表

**Request**

```
GET /v1/sets?id=xxx&mime_type=xxx
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 任务的id，可选|
| mime_type | string | 任务的文件类型，目前支持："image","video"，可选|


**Response**

```
200 OK
Content-Type: application/json

{
    "datas": [
        {
            "id": "xxx",
            "name": "xxx",
            "type": "monitor_active",
            "uri": "http://xxx",
            "monitor_interval": 30,
            "mime_types": ["image","video"],
            "cut_interval_msecs": 1000,
            "scenes": ["pulp","terror","politician"],
            "status": "running",
            "created_at": 1546394356,
            "modified_at": 1546394368
        },
        ...
    ]
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 任务的id|
| name | string | 任务的名称 |
| type | string | 任务的类型，取值有"monitor_active"、"monitor_passive"、"task"，其中task为文件任务 |
| scenes | list | 任务的审核场景，取值："pulp","terror","politician" |
| uri | string | 任务的api路径 |
| monitor_interval | int | 任务抓取数据的间隔时间 |
| mime_types | list | 任务的文件类型，取值："image","video" |
| cut_interval_msecs | int | 截帧频率，单位：毫秒 |
| status | string | 任务的状态，取值："running","stopped","completed" |
| created_at | int | 任务的创建时间，unix时间戳 |
| modified_at | int | 任务的最后一次修改时间，unix时间戳 |


### 1.7 获取任务历史

> 获取任务历史

**Request**

```
GET /v1/set/<id>/history
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 任务的id |


**Response**

```
200 OK
Content-Type: application/json

{
    "datas": [
        {
            "name": "xxx",
            "type": "monitor_active",
            "uri": "http://xxx",
            "monitor_interval": 30,
            "mime_types": ["image","video"],
            "cut_interval_msecs": 1000,
            "scenes": ["pulp","terror","politician"],
            "status": "running",
            "start_at": 1546394356,
            "end_at": 1546394368
        },
        ...
    ]
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| name | string | 任务的名称 |
| type | string | 任务的类型 |
| uri | string | 任务的api路径 |
| monitor_interval | int | 任务抓取数据的间隔时间 |
| mime_types | list | 任务的文件类型，取值："image","video" |
| cut_interval_msecs | int | 截帧频率，单位：毫秒 |
| scenes | list | 任务的审核场景，取值："pulp","terror","politician" |
| status | string | 任务的状态，取值："running","stopped","completed"。历史记录时该值为空 |
| start_at | int | 任务的开始时间，unix时间戳 |
| end_at | int | 任务的结束时间，unix时间戳 |


## 2. 审核管理

### 2.1 获取审核结果

> 获取审核结果

**Request**

```
POST /v1/censor/entries
Content-Type: application/json
{
    "set_id": "xxx",
    "suggestion": "block",
    "scenes": ["xxx"],
    "mime_type": "xxx",
    "start": xxx,
    "end": xxx,
    "marker": "xxx",
    "limit": xx
}

```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| set_id | string | 任务id，可选，默认从所有任务中取值 |
| suggestion | string | 审核结果类型：取值："block","review","pass"，可选，默认从所有结果类型中取值 |
| scenes | list | 审核场景类型列表：取值："pulp","terror","politician"，可选，默认从所有场景类型中取值 |
| mime_type | string | 文件类型，取值："image","video"，可选，默认从文件类型中取值 |
| start | int | 起始时间，unix时间戳，可选，start和end同时有值时则在[start, end]区间取值，否则从所有时间段取值 |
| end | int | 结束时间，unix时间戳，可选，start和end同时有值时则在[start, end]区间取值，否则从所有时间段取值 |
| marker | string | 上一次请求返回的标记，作为本次请求的起点信息，默认值为空字符串 |
| limit | int | 每页条数，可选，默认为20 |

**Response**

```
200 OK
Content-Type: application/json

{
    "total": 100,
    "marker": "xxxx",
    "datas": [
        {
            "id": "xxx",
            "set_id": "xxx",
            "uri": "http://xxx",
            "mime_type": "image",
            "cover_uri": "xxx",
            "cut_interval_msecs": 1000,
            "original": {
                "suggestion" : "block",
                "scenes" : {
                    "politician" : {
                        "suggestion" : "review",
                        "details" : [
                           {
                                "suggestion" : "review",
                                "label" : "xxx",
                                "group" : "xxx",
                                "score" : 0.991507828235626,
                                "detections" : [
                                    {
                                        "pts": [[205,400],[1213,400],[1213,535],[205,535]],
                                        "score":0.998
                                    }
                                ]
                            } 
                        ]
                    },
                    "pulp" : {
                        "suggestion" : "pass",
                        "details" : [ 
                            {
                                "suggestion" : "pass",
                                "label" : "normal",
                                "group" : "",
                                "score" : 0.999930202960968,
                                "detections" : []
                            }
                        ]
                    },
                    "terror" : {
                        "suggestion" : "block",
                        "details" : [ 
                            {
                                "suggestion" : "block",
                                "label" : "guns",
                                "group" : "",
                                "score" : 0.991507828235626,
                                "detections" : [
                                    {
                                        "pts": [[205,400],[1213,400],[1213,535],[205,535]],
                                        "score":0.998
                                    }
                                ]
                            }
                        ]
                    }
                }
            }],
            "final": null,
            "error": {
                "code" : 4000203,
                "message" : "fetch uri failed: http://xxx"
            },
            "created_at": 1546394356
        },
        ...
    ]
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :---| :----- | :----- |
| total | int | 符合条件的所有条目数，用于展示分页 |
| marker | string | 本次请求的标记信息，可作为下一次请求的参数传入，如果没有剩余条目则返回空字符串 |
| id | string | 审核数据的id |
| set_id | string | 审核数据所属的任务id |
| uri | string | 审核数据的uri |
| mime_type | string | 审核数据的类型 |
| cover_uri | string | 视频文件的封面路径 |
| cut_interval_msecs | int | 视频文件的截帧频率，单位：毫秒 |
| original | map | 审核结果，包括总的审核结果&各审核场景的审核结果。对于图片文件还包括详细标签信息（视频文件的详细信息通过视频帧结果接口获取） |
| final | map | 人审后的结果，未人审则为null |
| error | map | 审核错误，无错误则为null |
| error.code | string | 错误码 |
| error.message | string | 错误信息描述 |
| created_at | int | 审核时间，unix时间戳 |


错误码：

| 错误码 | 描述 |
| :--- | :--- |
| 4000100 | 请求参数错误 |
| 4000201 | 资源地址不支持 |
| 4000203 | 获取资源失败 |
| 4000204 | 获取资源超时 |
| 4150301 | 图片格式不支持 |
| 4000302 | 图片过大，图片长宽超过4999像素、或图片大小超过10M |
| 5000900 | 系统错误 |


鉴黄标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| pulp | 色情 |
| sexy | 性感 |
| normal | 正常 |


暴恐标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| illegal_flag | 违规旗帜 |
| knives | 刀 |
| guns | 枪 |
| anime_knives | 二次元刀 |
| anime_guns | 二次元枪 |
| bloodiness | 血腥 |
| self_burning | 自焚 |
| beheaded | 行刑斩首 |
| march_crowed | 非法集会 |
| fight_police | 警民冲突 |
| fight_person | 打架斗殴 |
| special_characters | 特殊字符 |
| anime_bloodiness | 二次元血腥 |
| special_clothing | 特殊着装 |
| normal | 正常 |


政治人物分组标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| domestic_statesman | 国内政治人物 |
| foreign_statesman | 国外政治人物 |
| affairs_official_gov | 落马官员（政府) |
| affairs_official_ent | 落马官员（企事业）|
| anti_china_people | 反华分子 |
| terrorist | 恐怖分子 |
| affairs_celebrity | 劣迹艺人 |
| chinese_martyr | 烈士 |


### 2.2 更改审核结果

> 更改审核结果

**Request**

```
POST /v1/censor/update/entries
Content-Type: application/json

{
    "ids": ["xx1","xx2"],
    "suggestion": "block"
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| ids | list | 欲更改的审核数据的id列表，必选 |
| suggestion | string | 人审结果，取值："block","review","pass"，必选 |


**Response**

```
200 OK
```


### 2.3 下载审核结果

> 下载审核结果

**Request**

```
POST /v1/censor/entries/download
Content-Type: application/json
{
    "set_id": "xxx",
    "suggestion": "block",
    "scenes": ["xxx"],
    "mime_type": "xxx",
    "start": xxx,
    "end": xxx
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| set_id | string | 任务id，可选，默认从所有任务中取值 |
| suggestion | string | 审核结果类型：取值："block","review","pass"，可选，默认从所有结果类型中取值 |
| scenes | list | 审核场景类型列表：取值："pulp","terror","politician"，可选，默认从所有场景类型中取值 |
| mime_type | string | 文件类型，取值："image","video"，可选，默认从文件类型中取值 |
| start | int | 起始时间，unix时间戳，可选，start和end同时有值时则在[start, end]区间取值，否则从所有时间段取值 |
| end | int | 结束时间，unix时间戳，可选，start和end同时有值时则在[start, end]区间取值，否则从所有时间段取值 |


**Response**

```
200 OK
Content-Type: application/octet-stream
***文件内容***
```


### 2.4 获取视频帧结果

> 获取视频帧结果

**Request**

```
GET /v1/censor/entry/<entry_id>/cuts?suggestion=xx&scene=xx&marker=xx&limit=xx
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| entry_id | string | 视频id |
| suggestion | string | 审核结果类型：取值："block","review","pass"，可选，默认从所有结果类型中取值 |
| scene | string | 审核场景类型：取值："pulp","terror","politician"，可选，默认从所有场景类型中取值 |
| marker | string | 上一次请求返回的标记，作为本次请求的起点信息，默认值为空字符串 |
| limit | int | 每页条数，可选，默认为20 |

**Response**

```
200 OK
Content-Type: application/json

{
    "total": 100,
    "marker": "xxxx",
    "datas": [
        {
            "id": "xxx",
            "offset": xxx,
            "uri": "xxx",
            "original": {
                "suggestion" : "block",
                "scenes" : {
                    "politician" : {
                        "suggestion" : "review",
                        "details" : [
                           {
                                "suggestion" : "review",
                                "label" : "xxx",
                                "group" : "xxx",
                                "score" : 0.991507828235626,
                                "detections" : [
                                    {
                                        "pts": [[205,400],[1213,400],[1213,535],[205,535]],
                                        "score":0.998
                                    }
                                ]
                            } 
                        ]
                    },
                    "pulp" : {
                        "suggestion" : "pass",
                        "details" : [ 
                            {
                                "suggestion" : "pass",
                                "label" : "normal",
                                "group" : "",
                                "score" : 0.999930202960968,
                                "detections" : []
                            }
                        ]
                    },
                    "terror" : {
                        "suggestion" : "block",
                        "details" : [ 
                            {
                                "suggestion" : "block",
                                "label" : "guns",
                                "group" : "",
                                "score" : 0.991507828235626,
                                "detections" : [
                                    {
                                        "pts": [[205,400],[1213,400],[1213,535],[205,535]],
                                        "score":0.998
                                    }
                                ]
                            }
                        ]
                    }
                }
            }]
        },
        ...
    ]
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :---| :----- | :----- |
| total | int | 符合条件的所有条目数，用于展示分页 |
| marker | string | 本次请求的标记信息，可作为下一次请求的参数传入，如果没有剩余条目则返回空字符串 |
| id | string | 帧id |
| uri | string | 帧图片路径 |
| offset | int | 帧时间位置，单位：毫秒 |
| original | map | 审核结果，包括帧的总审核结果&各审核场景的审核结果 |

## 3. 用户管理

### 3.1 新建用户

> 新建用户

**Request**

```
POST /v1/user/add
Content-Type: application/json

{
    "id": "xxx",
    "desc": "xxx",
    "password": "xxx",
    "roles": ["censor", "manage_set"]
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 用户id，必选 |
| desc | string | 用户描述，可选 |
| password | string | 用户密码，必选 |
| roles | list | 用户角色，取值："censor","manage_set"，必选 |


**Response**

```
200 OK
```


### 3.2 删除用户

> 删除用户

**Request**

```
POST /v1/user/delete
Content-Type: application/json

{
    "id": "xxx"
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 用户id，必选 |


**Response**

```
200 OK
```


### 3.3 更新用户

> 更新用户

**Request**

```
POST /v1/user/update
Content-Type: application/json

{
    "id": "xxx",
    "desc": "xxx",
    "roles": ["censor", "manage_set"]
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 用户id，必选 |
| desc | string | 用户描述，必选 |
| roles | list | 用户角色，取值："censor","manage_set"，必选 |


**Response**

```
200 OK
```


### 3.4 更新密码

> 更新密码

**Request**

```
POST /v1/user/password
Content-Type: application/json

{
    "old": "xxx",
    "new": "xxx",
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| old | string | 老密码，必选 |
| new | string | 新密码，必选 |

**Response**

```
200 OK
```


### 3.5 获取用户列表

> 获取用户列表。可传入keyword，则只返回id或desc包含keyword字段的用户

**Request**

```
GET /v1/users?keyword=xx
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| keyword | string | 搜索字段，可选 |

**Response**

```
200 OK
Content-Type: application/json

{
    "datas": [
        {
            "id": "xxx",
            "desc": "xxx",
            "roles": ["admin"],
            "created_at: 1546394356
        },
        {
            "id": "xxx",
            "desc": "xxx",
            "roles": ["censor", "manage_set"],
            "created_at: 1546394356
        },
        ...
    ]
}

返回字段说明：

| 字段 | 取值 | 说明 |
| :---| :----- | :----- |
| id | string | 用户id |
| desc | string | 用户描述 |
| roles | list | 用户角色，取值："admin","censor","manage_set" |
| created_at | int | 用户创建时间，unix时间戳 |
```


## 4. 登录相关

### 4.1 登录

> 登录

**Request**

```
POST /v1/login/
Content-Type: application/json

{
    "id": "xxx",
    "password": "xxx"
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| id | string | 用户id，必选 |
| password | string | 用户密码，必选 |


**Response**

```
200 OK
Content-Type: application/json

{
    "id": "xxx",
    "roles": ["admin"]
}

返回字段说明：

| 字段 | 取值 | 说明 |
| :---| :----- | :----- |
| id | string | 用户id，必选 |
| roles | list | 用户角色，取值："admin","censor","manage_set" |
```


### 4.2 登出

> 登出

**Request**

```
POST /v1/logout/
```

**Response**

```
200 OK
```


### 4.3 获取系统配置

> 获取系统配置，返回后台支持的场景、文件类型等信息，便于前端页面个性化展示

**Request**

```
GET /v1/config/
```

**Response**

```
200 OK
Content-Type: application/json

{
    "scenes": ["pulp","terror","politician"],
    "mime_types": ["image","video"]
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :---| :----- | :----- |
| scenes | list | 任务的审核场景，取值："pulp","terror","politician" |
| mime_types | list | 任务的文件类型，取值："image","video" |


## 5. 资源管理

### 5.1 批量上传资源

> 将资源上传至指定的任务中,任务必须为monitor_passive类型

**Request**

```
POST /v1/resources/{resource}
Content-Type: application/json

{
    "urls": ["xxx", "yyy"]
}
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| resource | string | 指定resource地址，必选 |
| urls | list | 资源URL数组，必选 |

**Response**

```
200 OK
```