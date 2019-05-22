<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1. 图片搜索](#1-%E5%9B%BE%E7%89%87%E6%90%9C%E7%B4%A2)
  - [1.1 创建 Group](#11-%E5%88%9B%E5%BB%BA-group)
  - [1.2 获取所有Group ID](#12-%E8%8E%B7%E5%8F%96%E6%89%80%E6%9C%89group-id)
  - [1.3 获取 Group 信息](#13-%E8%8E%B7%E5%8F%96-group-%E4%BF%A1%E6%81%AF)
  - [1.4 销毁 Group](#14-%E9%94%80%E6%AF%81-group)
  - [1.5 添加图片](#15-%E6%B7%BB%E5%8A%A0%E5%9B%BE%E7%89%87)
  - [1.6 检索图片](#16-%E6%A3%80%E7%B4%A2%E5%9B%BE%E7%89%87)
  - [1.7 删除图片](#17-%E5%88%A0%E9%99%A4%E5%9B%BE%E7%89%87)
  - [1.8 更新图片](#18-%E6%9B%B4%E6%96%B0%E5%9B%BE%E7%89%87)
  - [1.9 查询图片](#19-%E6%9F%A5%E8%AF%A2%E5%9B%BE%E7%89%87)
  - [1.10 获取Group Tag信息](#110-%E8%8E%B7%E5%8F%96group-tag%E4%BF%A1%E6%81%AF)
- [2. 人脸搜索](#2-%E4%BA%BA%E8%84%B8%E6%90%9C%E7%B4%A2)
  - [2.1 创建 Group](#21-%E5%88%9B%E5%BB%BA-group)
  - [2.2 获取所有Group ID](#22-%E8%8E%B7%E5%8F%96%E6%89%80%E6%9C%89group-id)
  - [2.3 获取 Group 信息](#23-%E8%8E%B7%E5%8F%96-group-%E4%BF%A1%E6%81%AF)
  - [2.4 销毁 Group](#24-%E9%94%80%E6%AF%81-group)
  - [2.5 人脸入库](#25-%E4%BA%BA%E8%84%B8%E5%85%A5%E5%BA%93)
  - [2.6 检索人脸](#26-%E6%A3%80%E7%B4%A2%E4%BA%BA%E8%84%B8)
  - [2.7 删除图片](#27-%E5%88%A0%E9%99%A4%E5%9B%BE%E7%89%87)
  - [2.8 更新图片](#28-%E6%9B%B4%E6%96%B0%E5%9B%BE%E7%89%87)
  - [2.9 查询图片](#29-%E6%9F%A5%E8%AF%A2%E5%9B%BE%E7%89%87)
  - [2.10 获取Group Tag信息](#210-%E8%8E%B7%E5%8F%96group-tag%E4%BF%A1%E6%81%AF)
  - [2.11 多库检索人脸](#211-%E5%A4%9A%E5%BA%93%E6%A3%80%E7%B4%A2%E4%BA%BA%E8%84%B8)
  - [2.12 单库聚类人脸](#212-%E5%8D%95%E5%BA%93%E8%81%9A%E7%B1%BB%E4%BA%BA%E8%84%B8)
  - [2.13. 人脸库比对](#213-%E4%BA%BA%E8%84%B8%E5%BA%93%E6%AF%94%E5%AF%B9)
  - [2.14 检索坐标人脸](#214-%E6%A3%80%E7%B4%A2%E5%9D%90%E6%A0%87%E4%BA%BA%E8%84%B8)
  - [2.15 多库检索坐标人脸](#215-%E5%A4%9A%E5%BA%93%E6%A3%80%E7%B4%A2%E5%9D%90%E6%A0%87%E4%BA%BA%E8%84%B8)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

基本语义说明
* 资源表示方式（URI）。通过统一方式定位、获取资源（图片、二进制数据等）
	* HTTP， 网络资源，形如：`http://host/path`、`https://host/path`
	* Data，Data URI Scheme形态的二进制文件，形如：`data:application/octet-stream;base64,xxx`。

# 1. 图片搜索

## 1.1 创建 Group

> 新建 Group，单次请求创建单个 Group

**Request**

```
POST /v1/image/groups/<group_name>
Content-Type: application/json

{
    "config:{
        "capacity": 300000
    }
}
```

**Response**

```
200 OK
```

接口字段说明：

| 字段       | 取值   | 说明                     |
| :--------- | :----- | :----------------------- |
| `group_name` | string | Group 唯一标识，长度 3-32 位，第一位必须是小写字母，其它位置是小写字母或者数字、下划线 、"-"          |
| `capacity`   | int    | Group 预期图片数目，必选，必须大于0，可以自动增长，实际限制受限于内存或者显存 |

## 1.2 获取所有Group ID
> 获取所有Group ID

**Request**

```
GET /v1/image/groups
```

**Response**

```
200 OK
Content-Type: application/json
{
    "groups": [
        "<group_id>",
        ...
    ]
}
```

接口字段说明：

| 字段       | 取值   | 说明                   |
| :--------- | :----- | :--------------------- |
| `groups`       | array  | 所有 group ID列表 |

## 1.3 获取 Group 信息

> 获取 Group 信息，单次请求单个 Group

**Request**

```
GET /v1/image/groups/<group_name>
```

**Response**

```
200 OK
Content-Type: application/json

{
    "config":{
        "capacity": 0
    },
    "tags": 1,
    "count":1
}
```

接口字段说明：

| 字段       | 取值   | 说明                   |
| :--------- | :----- | :--------------------- |
| `tags`       | int    | 当前 group 的 tag 数量 |
| `count`      | int    | 当前 group 的图片数目  |

## 1.4 销毁 Group

> 销毁 Group，单次请求删除单个 Group

**Request**

```
POST /v1/image/groups/<group_name>/remove
```

**Response**

```
200 OK
```

接口字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |



## 1.5 添加图片

> 添加图片到一个 Group

**Request**

```
POST /v1/image/groups/<group_name>/add
Content-Type: application/json

{
    "image":{
        "id": "",
        "uri": "",
        "tag": "",
        "desc": {}
    }
}
```

**Response**

```
200 OK

{
    "id":""
}
```

请求字段说明：

| 字段 | 取值   | 说明 |
| :---------- | :----- | :---------- |
| `image.id`   | string | 图片唯一标识，建议采用业务系统自己的图片 url 或者文件系统路径，搜索图片会返回此 id，如果id为空，则会随机生成ID返回 |
| `image.uri`  | string | 图片的uri, 必选，参见顶部语义说明  |
| `image.tag`  | string | 图片标签, 如果传空, 则默认为default |
| `image.desc` | object | 图片描述，可选，可以是布尔值、数字、字符串、数组和map，默认为空 |

## 1.6 检索图片

> 输入一张或者多张图片，返回与之相似的图片列表，按相似度排序

**Request**

```
POST /v1/image/groups/<group_name>/search
Content-Type: application/json

{
    "images":[
        "",
        ...
    ],
    "threshold": 0.9,
    "limit": 5
}
```

请求字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `images.[]` | string | 图片的uri, 参见顶部语义说明                      |
| `threshold` | float | 搜索图片阈值， 范围 [0,1]，推荐 0.3            |
| `limit`     | int | 返回的单张图片搜索到的图片数目限制，范围 [1,10000]， 默认 100 |
**Response**

```
200 OK
Content-Type: application/json

{
    "search_results": [
        {
            "results":[
                {
                    "id":"",
                    "score":0.9,
                    "tag":"",
                    "desc":{}
                }
            ]
        },
        ...
    ]
}
```

返回字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `search_results.results.id` | string | 图片唯一标识 |
| `search_results.results.score` | float | 图片搜索相似度，取值范围0~1，1为确定 |
| `search_results.results.tag` | string | 命中图片的标签 |
| `search_results.results.desc` | object | 命中图片的描述信息 |

## 1.7 删除图片

> 删除多张图片

**Request**

```
POST /v1/image/groups/<group_name>/delete
Content-Type: application/json

{
    "ids":[
        "AAAO054X4wzh",
        "AAAHdaa3wwzh"
    ]
}
```

请求字段说明：

| 字段       | 取值   | 说明                |
| :-----     | :----- | :------------------ |
| `ids.[]`     | string | 期望删除的图片 ID   |

**Response**

```
200 OK
Content-Type: application/json

{
    "deleted": [
        "AAAO054X4wzh",
        "AAAHdaa3wwzh"
    ]
}

```
返回字段说明：

| 字段       | 取值   | 说明                |
| :-----     | :----- | :------------------ |
| `deleted.[]` | string | 实际被删除的图片 ID |

## 1.8 更新图片

> 更新 id 对应的图片内容以及 tag、desc

**Request**

```
POST /v1/image/groups/<group_name>/update
Content-Type: application/json

{
    "image":{
        "id": "AAAO054X4wzh",
        "uri": "",
        "tag": "",
        "desc": {}
    }
}
```

请求字段说明：

| 字段 | 取值   | 说明 |
| :---------- | :----- | :---------- |
| `image.id`   | string | 图片唯一标识，建议采用业务系统自己的图片 url 或者文件系统路径，搜索图片会返回此 id，如果id为空，则会随机生成ID返回 |
| `image.uri`  | string | 图片的uri, 必选，参见顶部语义说明  |
| `image.tag`  | string | 图片标签, 如果传空, 则默认为default |
| `image.desc` | object | 图片描述，可选，可以是布尔值、数字、字符串、数组和map，默认为空 |

**Response**

```
200 OK
Content-Type: application/json
```

接口字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |


## 1.9 查询图片

> 列出 group 的所有图片，可以按照 tag 过滤

**Request**

```
GET /v1/image/groups/<group_name>/images?tag=<Tag>&marker=<Marker>&limit=<Limit>
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------------- |
| `tag` | string | 只查询指定 tag 的图片，可选 |
| `marker` | string | 上一次列举返回的位置标记，作为本次列举的起点信息。默认值为空字符串，可选。 |
| `limit` | int | 本次列举的条目数，范围为 1-1000。默认值为 1000。 |

**Response**

```
200 OK
Content-Type: application/json


{
    "images":[
        {
            "id": "AAAO054X4wzh",
            "tag": "",
            "desc": {}
        },
        ...
    ],
    "marker":"eyJjIjowLCJrIjoiMDAwMDAyLmljbyJ9"
}
```
返回字段说明：

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------------- |
| `image.id` | string | 图片唯一标识 |
| `image.tag`  | string | 图片标签 |
| `image.desc` | object | 图片描述 |
| `marker` | string | 有剩余条目则返回非空字符串，作为下一次列举的参数传入。如果没有剩余条目则返回空字符串 |

## 1.10 获取Group Tag信息

> 列出 group 的所有tag

**Request**

```
GET /v1/image/groups/<group_name>/tags?limit=<Limit>&marker=<Marker>
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------------- |
| `marker` | string | 上一次列举返回的位置标记，作为本次列举的起点信息。默认值为空字符串，可选。 |
| `limit` | int | 本次列举的条目数，范围为 1-1000。默认值为 1000。 |

**Response**

```
200 OK
Content-Type: application/json

{
    "config":{
        "capacity": 0
    },
    "tags": [
        {
            "name": "tag1",
            "count": 10
        },
        {
            "name": "tag2",
            "count": 10
        }
    ],
    "marker": "dGFnMQ=="
}
```
请求字段说明：

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------------- |
| `config.capacity`   | int    | Group 预期图片数目，必选，必须大于0，可以自动增长，实际限制受限于内存或者显存 |
| `tags.name` | string | 当前 tag 的名字 |
| `tags.count` | int    | 当前 tag 的图片数目  |
| `marker` | string | 有剩余条目则返回非空字符串，作为下一次列举的参数传入。如果没有剩余条目则返回空字符串。 |

# 2. 人脸搜索


## 2.1 创建 Group

> 新建 Group，单次请求创建单个 Group

**Request**

```
POST /v1/face/groups/<group_name>
Content-Type: application/json

{
    "config":{
        "capacity": 300000
    }
}
```

请求字段说明：

| 字段       | 取值   | 说明                     |
| :--------- | :----- | :----------------------- |
| `group_name` | string | Group 唯一标识，长度 3-32 位，第一位必须是小写字母，其它位置是小写字母或者数字、下划线 、"-"          |
| `config.capacity` | int | 预估人脸库大小，用于系统预分配资源，系统自动弹性扩大 |

**Response**

```
200 OK
```

## 2.2 获取所有Group ID
> 获取所有Group ID

**Request**

```
GET /v1/face/groups
```

**Response**

```
200 OK
Content-Type: application/json
{
    "groups": [
        "<group_id>",
        ...
    ]
}
```

返回字段说明：

| 字段       | 取值   | 说明                   |
| :--------- | :----- | :--------------------- |
| `groups.[]`       | string  | group ID列表 |

## 2.3 获取 Group 信息

> 获取 Group 信息，单次请求单个 Group

**Request**

```
GET /v1/face/groups/<group_name>
```

**Response**

```
200 OK
Content-Type: application/json

{
    "config":{
        "capacity": 0
    },
    "tags": 1,
    "count":1
}
```

返回字段说明：

| 字段       | 取值   | 说明                   |
| :--------- | :----- | :--------------------- |
| `config.capacity` | int | 预估人脸库大小 |
| `tags`       | int    | 当前 group 的 tag 数量 |
| `coun`      | int    | 当前 group 的图片数目  |

## 2.4 销毁 Group

> 销毁 Group，单次请求删除单个 Group

**Request**

```
POST /v1/face/groups/<group_name>/remove
```

**Response**

```
200 OK
```

接口字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |

## 2.5 人脸入库

> 添加人脸到一个 Group

**Request**

```
POST /v1/face/groups/<group_name>/add
Content-Type: application/json

{
    "image":{
        "id": "",
        "uri": "",
        "tag": "",
        "desc": {},
        "bounding_box":{
            "pts":[[100,200],[300,400],[400,500],[500,600]],
        }
    },
    "params": {
        "reject_bad_face": true
    }
}
```
请求字段说明：

| 字段        | 取值   | 说明 |
| :---------- | :----- | :------ |
| `image.id`   | string | 图片唯一标识，建议采用业务系统自己的图片 url 或者文件系统路径，搜索图片会返回此 id，如果id为空，则会随机生成ID返回 |
| `image.uri`  | string | 图片的uri, 必选，参见顶部语义说明 |
| `image.tag`  | string | 图片标签, 如果传空, 则默认为default |
| `image.desc` |  object | 图片描述，可选，可以是布尔值、数字、字符串、数组和map，默认为空  |
| `image.bounding_box.pts` | array | 图片人脸位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部|
| `params.reject_bad_face` | bool | 是否拒绝低质量非清晰人脸入库，默认为false，即所示输入的合格人脸均入库 |

**注意**

* reject_bad_face=true时，输入低质量人脸会返回400错误；reject_bad_face=false时，输入低质量人脸会返回200，人脸正确入库，但会返回人脸质量信息
* reject_bad_face=true时人脸质量判定策略为：quality=clear && orientation=up允许入库；其他均拒绝

**Response**

```
200 OK

{
    "id":"xxxxx",
    "bounding_box": {
        "pts": [[100,200],[300,400],[400,500],[500,600]],
        "score": 0.998
    },
    "face_quality": {
        "quality": "blur",
        "orientation": "up_left"
    },
    "face_number": 2
}

```

返回字段说明

| 字段        | 取值   | 说明 |
| :---------- | :----- | :------ |
| `id`   | string | 图片ID |
| `bounding_box.pts`  | array | 人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `bounding_box.score` | float | 人脸的检测置信度，请求中指定坐标框时无该字段返回 |
| `face_quality` | object | 人脸质量评估结果，请求中指定坐标框时无该字段返回 |
| `face_quality.quality` | string | 人脸质量分类，目前类别为"clear"、"blur"、"small"、"cover"、"pose"，分别对应"清晰"、"模糊"、"过小人脸"、"人脸覆盖"、"人脸大姿态" |
| `face_quality.orientation` | string | 人脸方向，指的人脸中轴线的旋转角度，目前有"up"、"up_left"、"left"、"down_left"、"down"、"down_right"、"right"、"up_right" |
| `face_number` | int | 检出人脸数，静态配置multi_faces_mode=1时且图片实际人脸多于1时会返回图片实际检出人脸数 |

## 2.6 检索人脸

> 输入一张或者多张包含人脸的图片，对于每张图片里面的每张人脸，会返回与这张人脸最接近的人脸

**Request**

```
POST /v1/face/groups/<group_name>/search
Content-Type: application/json

{
    "images":[
        "",
        ...
    ],
    "threshold": 0.45,
    "limit": 5,
    "use_quality": true //非公开参数
}
```

请求字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `images.[]` | string | 图片的uri, 参见顶部语义说明                      |
| `threshold` | float | 搜索图片阈值， 范围 [0,1]，推荐 图片0.45、监控0.35         |
| `limit`     | int | 返回的单张图片搜索到的图片数目限制，范围 [1,10000]， 默认 100 |
| `use_quality` | bool | 是否根据质量评估过滤待搜索人脸，use_quality=true时，仅搜索清晰(clear)、朝上(up)的人脸，默认use_quality=false |

**Response**

```
200 OK
Content-Type: application/json

{
    "search_results": [
        {
            "faces":[
                {
                    "bounding_box":{
                        "pts":[[100,200],[300,400],[400,500],[500,600]],
                        "score": 0.9
                    },
                    "faces":[
                        {
                            "score"0.9
                            "id":"",
                            "tag":"",
                            "desc":{},
                            "bounding_box": {
                                "pts": [[100,200],[300,400],[400,500],[500,600]],
                                "score": 0.998
                            }
                        }
                    ]
                }
            ]
        },
        ...
    ]
}
```


返回字段说明:

| 字段      | 取值   | 说明 |
| :-------- | :----- | :---------- |
| `search_results`     | array | 输入图片里面每一张图片的结果对应这个数组的一项 |
| `search_results.[].faces`     | array | 某张图片里面检测到的人脸 |
| `search_results.[].faces.[].bounding_box`     | object | 某张图片里面检测到人脸的坐标 |
| `search_results.[].faces.[].bounding_box.pts`     | array | 人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.[].bounding_box.score`     | number | 人脸的检测置信度 |
| `search_results.[].faces.faces`     | object | 和此人脸相似的人脸的列表 |
| `search_results.[].faces.faces.id`  | string | 图片唯一标识 |
| `search_results.[].faces.faces.tag` | string | 图片标签信息 |
| `search_results.[].faces.faces.desc` | object | 图片描述 |
| `search_results.[].faces.faces.score`     | float | 两个人脸的相似度，取值范围0~1，1为确定 |
| `search_results.[].faces.faces.bounding_box.pts`     | array | 底库图片中的人脸位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.faces.bounding_box.score`     | float | 底库图片中人脸的检测置信度 |

## 2.7 删除图片

> 删除多张图片

**Request**

```
POST /v1/face/groups/<group_name>/delete
Content-Type: application/json

{
    "ids":[
        "AAAO054X4wzh",
        "AAAHdaa3wwzh"
    ]
}
```

请求字段说明：

| 字段   | 取值   | 说明              |
| :----- | :----- | :---------------- |
| `ids.[]` | string | 期望删除的图片 ID |
**Response**

```
200 OK
Content-Type: application/json

{
    "deleted": [
        "AAAO054X4wzh",
        "AAAHdaa3wwzh"
    ]
}

```
返回字段说明：

| 字段   | 取值   | 说明              |
| :----- | :----- | :---------------- |
| `deleted.[]` | string | 实际被删除的图片 ID |

## 2.8 更新图片

> 更新 id 对应的图片内容以及 tag、desc

**Request**

```
POST /v1/face/groups/<group_name>/update
Content-Type: application/json

{
    "image":{
        "id": "AAAO054X4wzh",
        "uri": "",
        "tag": "",
        "desc": {}
    }
}
```
请求字段说明：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| `image.id` | string | 图片唯一标识 |
| `image.uri`  | string | 图片的uri, 参见顶部语义说明 |
| `image.tag`  | string | 图片标签 |
| `image.desc` | object | 图片描述，可选，可以是布尔值、数字、字符串、数组和map，默认为空 |

**Response**

```
200 OK
Content-Type: application/json
```



## 2.9 查询图片

> 列出 group 的所有图片，可以按照 tag 过滤

**Request**

```
GET /v1/face/groups/<group_name>/images?tag=<Tag>&marker=<Marker>&limit=<Limit>
```

请求字段说明:

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------------- |
| `tag` | string | 只查询指定 tag 的图片，可选 |
| `marker` | string | 上一次列举返回的位置标记，作为本次列举的起点信息。默认值为空字符串，可选。 |
| `limit` | int | 本次列举的条目数，范围为 1-1000。默认值为 1000。 |

**Response**

```
200 OK
Content-Type: application/json


{
    "images":[
        {
            "id": "AAAO054X4wzh",
            "tag": "",
            "desc": {},
            "bounding_box": {
                "pts": [[100,200],[300,400],[400,500],[500,600]],
                "score": 0.998
            }
        },
        ...
    ],
    "marker":"eyJjIjowLCJrIjoiMDAwMDAyLmljbyJ9"
}
```


返回字段说明:

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------- |
| `images.id` | string | 图片唯一标识 |
| `images.tag` | string | 图片标签信息 |
| `images.desc` | object | 图片描述，可选，可以是布尔值、数字、字符串、数组和map，默认为空 |
| `images.bounding_box.pts`  | array | 人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `images.bounding_box.score` | float | 人脸的检测置信度 |
| `marker` | string | 有剩余条目则返回非空字符串，作为下一次列举的参数传入。如果没有剩余条目则返回空字符串。 |

## 2.10 获取Group Tag信息

> 列出 group 的所有tag

**Request**

```
GET /v1/face/groups/<group_name>/tags?limit=<Limit>&marker=<Marker>
```

请求字段说明：

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------------- |
| `marker` | string | 上一次列举返回的位置标记，作为本次列举的起点信息。默认值为空字符串，可选。 |
| `limit` | int | 本次列举的条目数，范围为 1-1000。默认值为 1000。 |

**Response**

```
200 OK
Content-Type: application/json

{
    "config":{
        "capacity": 0
    },
    "tags": [
        {
            "name": "tag1",
            "count": 10
        },
        {
            "name": "tag2",
            "count": 10
        }
    ],
    "marker": "dGFnMQ=="
}
```

返回字段说明：

| 字段 | 取值 | 说明 |
| :----- | :----- | :---------------- |
| `tags.name`   | string | 当前 tag 的名字 |
| `tags.count` | int    | 当前 tag 的图片数目  |
| `marker` | string | 有剩余条目则返回非空字符串，作为下一次列举的参数传入。如果没有剩余条目则返回空字符串。 |

## 2.11 多库检索人脸

> 输入一张或者多张包含人脸的图片，对于每张图片里面的每张人脸，会在多个库里面返回与这张人脸最接近的人脸,若使用cluster_group则库有相似人脸则输出，否则入聚类库

**Request**

```
POST /v1/face/groups/multi/search
Content-Type: application/json

{
    "images":[
        "",
        ...
    ],
    "groups":[
        "",
        ...
    ],
    "cluster_group":"",
    "threshold": 0.45,
    "limit": 5,
    "use_quality": true //非公开参数
}
```

请求字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `images.[]` | string | 图片的uri, 参见顶部语义说明                      |
| `groups.[]` | string | 图片所在组, 最多处理去重后5个不同的组名                   |
| `cluster_group` | string | 图片所在聚类库，底库不匹配的人脸入此库     |
| `threshold` | float | 搜索图片阈值， 范围 [0,1]，推荐 图片0.45、监控0.35           |
| `limit`     | int | 返回的单张图片搜索到的图片数目限制，范围 [1,10000]， 默认 100 |
| `use_quality` | bool | 是否根据质量评估过滤待搜索人脸，use_quality=true时，仅搜索清晰(clear)、朝上(up)的人脸，默认use_quality=false |

**Response**

```
200 OK
Content-Type: application/json

{
    "search_results": [
        {
            "faces":[
                {
                    "bounding_box":{
                        "pts":[[100,200],[300,400],[400,500],[500,600]],
                        "score": 0.9
                    },
                    "faces":[
                        {
                            "score"0.9
                            "id":"",
                            "tag":"",
                            "desc":{},
                            "group":"",
                            "bounding_box": {
                                "pts": [[100,200],[300,400],[400,500],[500,600]],
                                "score": 0.998
                            }
                        }
                    ]
                }
            ]
        },
        ...
    ]
}
```

返回字段说明:

| 字段      | 取值   | 说明 |
| :-------- | :----- | :---------- |
| `search_results`     | array | 输入图片里面每一张图片的结果对应这个数组的一项 |
| `search_results.[].faces`     | array | 某张图片里面检测到的人脸 |
| `search_results.[].faces.[].bounding_box`     | object | 某张图片里面检测到人脸的坐标 |
| `search_results.[].faces.[].bounding_box.pts`     | array | 人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.[].bounding_box.score`     | number | 人脸的检测置信度 |
| `search_results.[].faces.faces`     | object | 和此人脸相似的人脸的列表 |
| `search_results.[].faces.faces.id`  | string | 图片唯一标识 |
| `search_results.[].faces.faces.tag` | string | 图片标签信息 |
| `search_results.[].faces.faces.desc` | object | 图片描述 |
| `search_results.[].faces.faces.group` | string | 图片所在组 |
| `search_results.[].faces.faces.score`     | float | 两个人脸的相似度，取值范围0~1，为1时，即底库没有匹配的人脸，该人脸入库|
| `search_results.[].faces.faces.bounding_box.pts`     | array | 底库图片中的人脸位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.faces.bounding_box.score`     | float | 底库图片中人脸的检测置信度 |

## 2.12 单库聚类人脸

> 输入一张或者多张包含人脸的图片，对于每张图片里面的每张人脸，若库有相似人脸则输出，否则入聚类库

**Request**

```
POST /v1/face/groups/<group_name>/cluster
Content-Type: application/json

{
    "images":[
        "",
        ...
    ],
    "threshold": 0.45,
    "limit": 5,
    "use_quality": true //非公开参数
}
```

请求字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `images.[]` | string | 图片的uri, 参见顶部语义说明                      |
| `threshold` | float | 搜索图片阈值， 范围 [0,1]，推荐 图片0.45、监控0.35         |
| `limit`     | int | 返回的单张图片搜索到的图片数目限制，范围 [1,10000]， 默认 100 |
| `use_quality` | bool | 是否根据质量评估过滤待搜索人脸，use_quality=true时，仅搜索清晰(clear)、朝上(up)的人脸，默认use_quality=false |

**Response**

```
200 OK
Content-Type: application/json

{
    "search_results": [
        {
            "faces":[
                {
                    "bounding_box":{
                        "pts":[[100,200],[300,400],[400,500],[500,600]],
                        "score": 0.9
                    },
                    "faces":[
                        {
                            "score"0.9
                            "id":"",
                            "tag":"",
                            "desc":{},
                            "bounding_box": {
                                "pts": [[100,200],[300,400],[400,500],[500,600]],
                                "score": 0.998
                            }
                        }
                    ]
                }
            ]
        },
        ...
    ]
}
```

返回字段说明:

| 字段      | 取值   | 说明 |
| :-------- | :----- | :---------- |
| `search_results`     | array | 输入图片里面每一张图片的结果对应这个数组的一项 |
| `search_results.[].faces`     | array | 某张图片里面检测到的人脸 |
| `search_results.[].faces.[].bounding_box`     | object | 某张图片里面检测到人脸的坐标 |
| `search_results.[].faces.[].bounding_box.pts`     | array | 人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.[].bounding_box.score`     | number | 人脸的检测置信度 |
| `search_results.[].faces.faces`     | object | 和此人脸相似的人脸的列表 |
| `search_results.[].faces.faces.id`  | string | 图片唯一标识 |
| `search_results.[].faces.faces.tag` | string | 图片标签信息 |
| `search_results.[].faces.faces.desc` | object | 图片描述 |
| `search_results.[].faces.faces.score`  | float | 两个人脸的相似度，取值范围0~1，为1时，即底库没有匹配的人脸，该人脸入库 |
| `search_results.[].faces.faces.bounding_box.pts`     | array | 底库图片中的人脸位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.faces.bounding_box.score`     | float | 底库图片中人脸的检测置信度 |


## 2.13. 人脸库比对
> 已创建并完成人脸入库的两个人脸库进行比对，即A库（目标库）中所有人脸与B库（底库）中所有人脸进行比对，返回超过阈值的B库中对应的Top N命中人脸

**Request**

```
POST /v1/face/groups/<group_id>/compare
Content-Type: application/json

{
    "target_group": <target_group_id>,
    "threshold": 0.45,
    "limit": 5
}
```

请求字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `group_id` | string | 底库ID    |
| `target_group` | string | 目标库ID   |
| `threshold` | float | 搜索图片阈值， 范围 [0,1]，推荐 图片0.45、监控0.35        |
| `limit`     | int | 返回的单张图片搜索到的图片数目限制，范围 [1,10000]， 默认 100 |

**Response**

```
200 OK 

{
    "compare_results": [       
        {
            "id": "",
            "tag": "",
            "desc":{},
            "bounding_box":{
                "pts":[[100,200],[300,400],[400,500],[500,600]],
                "score": 0.9
            },
            "faces":[
                {
                    "score"0.9
                    "id":"",
                    "tag":"",
                    "desc":{},
                    "bounding_box": {
                        "pts": [[100,200],[300,400],[400,500],[500,600]],
                        "score": 0.998
                    }
                },
                ...
            ]
        },
        ...
    ]
}
```

返回字段说明:

| 字段      | 取值   | 说明 |
| :-------- | :----- | :---------- |
| `compare_results`     | array | 目标库比对结果数组 |
| `compare_results.[].id`     | string | 目标库人脸唯一标识  |
| `compare_results.[].tag` | string | 目标库人脸标签信息 |
| `compare_results.[].desc` | object | 目标库人脸描述 |
| `compare_results.[].bounding_box`     | object | 目标库人脸的坐标 |
| `compare_results.[].bounding_box.pts`     | array | 目标库人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `compare_results.[].bounding_box.score`     | number | 目标库人脸的检测置信度 |
| `compare_results.[].faces`     | object | 与目标库人脸相似的底库人脸的列表 |
| `compare_results.[].faces.id`  | string | 底库人脸唯一标识 |
| `compare_results.[].faces.tag` | string | 底库人脸标签信息 |
| `compare_results.[].faces.desc` | object | 底库人脸描述 |
| `compare_results.[].faces.score`     | float | 两个人脸的相似度，取值范围0~1，1为确定 |
| `compare_results.[].faces.bounding_box.pts`     | array | 底库人脸位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `compare_results.[].faces.bounding_box.score`     | float | 底库人脸的检测置信度 |


## 2.14 检索坐标人脸

> 输入一张或者多张包含人脸的图片，并输入每张图片里面的每张人脸的坐标框，会返回与这张人脸最接近的人脸

**Request**

```
POST /v1/face/groups/<group_name>/pts/search
Content-Type: application/json

{
    
    "data":[{
    	"uri":"http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
    	"attribute":{
    		"bounding_boxes":[{
    			"pts":[
                    [
                        52,
                        74
                    ],
                    [
                        215,
                        74
                    ],
                    [
                        215,
                        272
                    ],
                    [
                        52,
                        272
                    ]
                ]
    		}]
    	}
    }],
    "params":{
    	"threshold": 0.45,
    	"limit": 5
    }
    
}

```

请求字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `data.[].uri` | string | 图片的uri, 参见顶部语义说明                      |
| `data.[].attribute.bounding_boxes.[].pts` | int | 对应图片的坐标框      |
| `params.limit`     | int | 返回的单张图片搜索到的图片数目限制，范围 [1,10000]， 默认 100 |
| `params.threshold` | float | 搜索图片阈值， 范围 [0,1]，推荐 图片0.45、监控0.35  |


**Response**

```
200 OK
Content-Type: application/json

{
    "search_results": [
        {
            "faces":[
                {
                    "bounding_box":{
                        "pts":[[100,200],[300,400],[400,500],[500,600]],
                    },
                    "faces":[
                        {
                            "score"0.9
                            "id":"",
                            "tag":"",
                            "desc":{},
                            "bounding_box": {
                                "pts": [[100,200],[300,400],[400,500],[500,600]],
                                "score": 0.998
                            }
                        }
                    ]
                }
            ]
        },
        ...
    ]
}
```


返回字段说明:

| 字段      | 取值   | 说明 |
| :-------- | :----- | :---------- |
| `search_results`     | array | 输入图片里面每一张图片的结果对应这个数组的一项 |
| `search_results.[].faces`     | array | 某张图片里面检测到的人脸 |
| `search_results.[].faces.[].bounding_box`     | object | 某张图片里面检测到人脸的坐标 |
| `search_results.[].faces.[].bounding_box.pts`     | array | 人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.faces`     | object | 和此人脸相似的人脸的列表 |
| `search_results.[].faces.faces.id`  | string | 图片唯一标识 |
| `search_results.[].faces.faces.tag` | string | 图片标签信息 |
| `search_results.[].faces.faces.desc` | object | 图片描述 |
| `search_results.[].faces.faces.score`     | float | 两个人脸的相似度，取值范围0~1，1为确定 |
| `search_results.[].faces.faces.bounding_box.pts`     | array | 底库图片中的人脸位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.faces.bounding_box.score`   | float | 底库图片中人脸的检测置信度 |


## 2.15 多库检索坐标人脸

> 输入一张或者多张包含人脸的图片，并输入每张图片里面的每张人脸的坐标框，对于每张图片里面的每张人脸，会在多个库里面返回与这张人脸最接近的人脸,若使用cluster_group则库有相似人脸则输出，否则入聚类库

**Request**

```
POST /v1/face/groups/multi/pts/search
Content-Type: application/json

{
   "data":[{
    	"uri":"http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
    	"attribute":{
    		"bounding_boxes":[{
    			"pts":[
                    [
                        52,
                        74
                    ],
                    [
                        215,
                        74
                    ],
                    [
                        215,
                        272
                    ],
                    [
                        52,
                        272
                    ]
                ]
    		}]
    	}
    }],
    "params":{
    	"threshold": 0.45,
    	"limit": 5
    },
    "cluster_group":"",
    "groups":[
   	"trots",
   	"tro_face"
   	]
}
```

请求字段说明：

| 字段      | 取值   | 说明                                           |
| :-------- | :----- | :--------------------------------------------- |
| `data.[].uri` | string | 图片的uri, 参见顶部语义说明                      |
| `data.[].attribute.bounding_boxes.[].pts` | int | 对应图片的坐标框      |
| `groups.[]` | string | 图片所在组, 最多处理去重后5个不同的组名                   |
| `cluster_group` | string | 图片所在聚类库，底库不匹配的人脸入此库     |
| `params.threshold` | float | 搜索图片阈值， 范围 [0,1]，推荐 图片0.45、监控0.35           |
| `params.limit`     | int | 返回的单张图片搜索到的图片数目限制，范围 [1,10000]， 默认 100 |


**Response**

```
200 OK
Content-Type: application/json

{
    "search_results": [
        {
            "faces":[
                {
                    "bounding_box":{
                        "pts":[[100,200],[300,400],[400,500],[500,600]],
                    },
                    "faces":[
                        {
                            "score"0.9
                            "id":"",
                            "tag":"",
                            "desc":{},
                            "group":"",
                            "bounding_box": {
                                "pts": [[100,200],[300,400],[400,500],[500,600]],
                                "score": 0.998
                            }
                        }
                    ]
                }
            ]
        },
        ...
    ]
}
```

返回字段说明:

| 字段      | 取值   | 说明 |
| :-------- | :----- | :---------- |
| `search_results`     | array | 输入图片里面每一张图片的结果对应这个数组的一项 |
| `search_results.[].faces`     | array | 某张图片里面检测到的人脸 |
| `search_results.[].faces.[].bounding_box`     | object | 某张图片里面检测到人脸的坐标 |
| `search_results.[].faces.[].bounding_box.pts`     | array | 人脸所在图片中的位置，四点坐标值 ，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.faces`     | object | 和此人脸相似的人脸的列表 |
| `search_results.[].faces.faces.id`  | string | 图片唯一标识 |
| `search_results.[].faces.faces.tag` | string | 图片标签信息 |
| `search_results.[].faces.faces.desc` | object | 图片描述 |
| `search_results.[].faces.faces.group` | string | 图片所在组 |
| `search_results.[].faces.faces.score`     | float | 两个人脸的相似度，取值范围0~1，为1时，即底库没有匹配的人脸，该人脸入库|
| `search_results.[].faces.faces.bounding_box.pts`     | array | 底库图片中的人脸位置，四点坐标值，[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `search_results.[].faces.faces.bounding_box.score`     | float | 底库图片中人脸的检测置信度 |