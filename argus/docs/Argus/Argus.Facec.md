<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [人脸聚类](#%E4%BA%BA%E8%84%B8%E8%81%9A%E7%B1%BB)
  - [API](#api)
    - [/v1/face/cluster](#v1facecluster)
      - [请求](#%E8%AF%B7%E6%B1%82)
      - [参数说明](#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)
      - [返回](#%E8%BF%94%E5%9B%9E)
      - [结果说明](#%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E)
    - [/v1/face/cluster/gid](#v1faceclustergid)
      - [请求](#%E8%AF%B7%E6%B1%82-1)
      - [参数说明](#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E-1)
      - [返回](#%E8%BF%94%E5%9B%9E-1)
      - [结果说明](#%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E-1)
    - [/v1/face/cluster/gather](#v1faceclustergather)
      - [请求](#%E8%AF%B7%E6%B1%82-2)
      - [参数说明](#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E-2)
      - [返回](#%E8%BF%94%E5%9B%9E-2)
    - [/v1/face/cluster/adjust](#v1faceclusteradjust)
      - [请求](#%E8%AF%B7%E6%B1%82-3)
      - [参数说明](#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E-3)
      - [返回](#%E8%BF%94%E5%9B%9E-3)
    - [/v1/face/cluster/merge](#v1faceclustermerge)
      - [请求](#%E8%AF%B7%E6%B1%82-4)
      - [参数说明](#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E-4)
      - [返回](#%E8%BF%94%E5%9B%9E-4)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# 人脸聚类

## API

| PATH | METHOD | 说明 |
| :--- | :--- | :--- |
| [GET /v1/face/cluster](#/v1/face/cluster) | GET | 获取聚类的结果 |
| [GET /v1/face/cluster/gid](#/v1/face/cluster/gid) | GET | 获取指定分组的结果 |
| [POST /v1/face/cluster/gather](#/v1/face/cluster/gather) | POST | 发起图片的人脸聚类检测 |
| [POST /v1/facec/cluster/adjust](#/v1/face/cluster/adjust) | POST | 手动调整人脸分组 |
| [POST /v1/facec/cluster/merge](#/v1/face/cluster/merge) | POST | 手动合并人脸分组 |

### /v1/face/cluster

获得当前终端用户的人脸聚类结果，返回分组列表和每个聚类组的概要。

#### 请求

```
GET /v1/face/cluster?euid=<euid>&include_uncategory=<false|true> HTTP/1.1
Authorization: Qiniu <AccessKey>:<Sign>
```

#### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `euid` | string | Y | 终端用户唯一标识 |
| `include_uncategory` | bool | X | 是否包含隐含分组 |

#### 返回

```
200 OK
Content-Type: application/json

{
    "groups": [
        {
            "id": <gid>,
            "face_count": <face_count>,
            "faces": [
                {
                    "uri": <image_uri>,
                    "pts": [[476, 90], [537, 90], [537, 135], [476, 135]],
                    "score": 0.88
                },
                ...
            ]
        },
        ...
    ]
}
```

#### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | :--- |
| `gid` | int | 分组的唯一标识，一般为正整数；<0有特殊含义 |
| `face_count` | int | 分组中人脸的统计值 |
| `faces.uri` | string | 人脸对应的图片地址 |
| `faces.pts` | array | 人脸所在图片中的位置 |
| `faces.score` | float | 人脸的检测置信度 |


### /v1/face/cluster/gid

获得当前终端用户指定分组的人脸聚类结果

#### 请求

```
GET /v1/face/cluster/<gid>?euid=<euid> HTTP/1.1
Authorization: Qiniu <AccessKey>:<Sign>
```

#### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `euid` | string | Y | 终端用户唯一标识 |
| `gid` | int | Y | 指定分组的唯一标识 |

#### 返回

```
200 OK
Content-Type: application/json

{
    "face_count": <face_count>,
    "faces": [
        {
            "uri": <image_uri>,
            "pts": [[476, 90], [537, 90], [537, 135], [476, 135]],
            "score": 0.88
        },
        ...
    ]
}
```

#### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | :--- |
| `face_count` | int | 分组中人脸的统计值 |
| `faces.uri` | string | 人脸对应的图片地址 |
| `faces.pts` | array | 人脸所在图片中的位置 |
| `faces.score` | float | 人脸的检测置信度 |


### /v1/face/cluster/gather

发起图片的人脸聚类检测。

#### 请求

```
POST /v1/face/cluster/gather HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "euid": <euid>,
    "items": [
        <image_uri>,
        ...
    ]
}
```

#### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `euid` | string | Y | 终端用户唯一标识 |
| `image_uri` | string | Y | 图片地址 |

#### 返回

```
200 OK
```


### /v1/face/cluster/adjust

手动调整人脸分组。

#### 请求

```
POST /v1/face/cluster/adjust HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "euid": <euid>,
    "from_group_id": <from_group_id>,
    "to_group_id": <to_group_id>,
    "items": [
        <image_uri>,
        ...
    ]
}
```

#### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `euid` | string | Y | 终端用户唯一标识 |
| `from_group_id` | int | Y | 分组的唯一标识 |
| `to_group_id` | int | Y | 分组的唯一标识 |
| `image_uri` | string | Y | 图片地址 |

#### 返回

```
200 OK
```


### /v1/face/cluster/merge

手动调整人脸分组。

#### 请求

```
POST /v1/face/cluster/merge HTTP/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
    "euid": <euid>,
    "to_group_id": <to_group_id>,
    "from_groups": [
        <gid>,
        ...
    ]
}
```

#### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `euid` | string | Y | 终端用户唯一标识 |
| `to_group_id` | int | Y | 分组的唯一标识 |
| `from_groups.gid` | int | Y | 分组的唯一标识 |

#### 返回

```
200 OK
```
