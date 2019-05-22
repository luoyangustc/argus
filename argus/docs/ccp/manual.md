<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

PostSets
- [Admin API]
  - [资源实体](#资源实体)
	- [LabelTitle](#LabelTitle)
  - [API](#api)
    - [Argus-Manual](#argus-manual)
      - [POST/v1/ccp/manual/sets](#post/v1/ccp/manual/sets)
      - [GET/v1/ccp/manual/sets/<set_id>](#get/v1/ccp/manual/sets/<set_id>)
      - [GET/v1/ccp/manual/sets](#get/v1/ccp/manual/sets)
      - [POST/v1/ccp/manual/sets/<set_id>/entries](#post/v1/ccp/manual/sets/<set_id>/entries)

# API
## post/v1/ccp/manual/sets
> 创建规则

### 请求

```
POST /v1/ccp/manual/sets

{
    "set_id":<set_id>",
    "uid":<uid>,
    "source_type":"KODO",
    "type":"batch",
    "image":{
        "is_on":true,
        "scenes":[
            "pulp"
        ]
    },
    "video":{
        "is_on":true,
        "scenes":[
            "pulp"
        ]
    },
    "notify_url":"http://notify_url.com"
}
```

### 参数说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `set_id` | string | 规则的id |
| `uid` | uint32 | 用户的uid |
| `source_type` | string | 资源的来源: "KODO", "API" |
| `type` | string | 资源的处理方式: "stream","batch"  |
| `image.is_on` | bool | 是否是图片审核 |
| `image.scenes` | []string | 图片审核的类型:"pulp","terror", "politician",... |
| `video.is_on` | bool | 是否是视频审核  |
| `video.scenes` | []string | 视频审核的类型:"pulp","terror", "politician",... |
| `notify_url` | string | 机审的回调地址 |

### 返回

```
200 OK
```

## get/v1/ccp/manual/sets
> 得到所有规则

### 请求

```
GET /v1/ccp/manual/sets
```

### 返回

```
200 OK
{
    "result":[
        {
            "set_id":"20180723131731",
            "source_type":"KODO",
            "type":"batch",
            "image":{
                "is_on":true,
                "scenes":[
                    "pulp"
                ]
            },
            "video":{
                "is_on":true,
                "scenes":[
                    "pulp"
                ]
            },
            "notify_url":"http://argus-bcp.xxx/v1/cap/jobs/*/done"
        },
        ...
    ]
}
```

## get/v1/ccp/manual/sets/<set_id>
> 得到单个规则

### 请求

```
GET /v1/ccp/manual/sets/<set_id>

```

### 返回

```
200 OK
{
    "set_id":"20180723131731",
    "source_type":"KODO",
    "type":"batch",
    "image":{
        "is_on":true,
        "scenes":[
            "pulp"
        ]
    },
    "video":{
        "is_on":true,
        "scenes":[
            "pulp"
        ]
    },
    "notify_url":"http://argus-bcp.xxx/v1/cap/jobs/*/done"
}
```

## post/v1/ccp/manual/sets/<set_id>/entries
> 创建批量人审任务

### 请求

```
POST /v1/ccp/manual/sets/<set_id>/entries
{
    "uid":<uid>,
    "bucket":<user_bucket>,
    "keys":<keys>
}

```
### 参数说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `uid` | uint32 | 用户的id |
| `bucket` | string | 存储要审核的图片文件的bucket |
| `keys` | []string | 要审核的图片文件 |

### 返回

```
200 OK
```