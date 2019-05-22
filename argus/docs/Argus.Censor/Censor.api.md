<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [API概览](#api%E6%A6%82%E8%A7%88)
  - [提交任务参数说明](#%E6%8F%90%E4%BA%A4%E4%BB%BB%E5%8A%A1%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)
  - [识别结果说明](#%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E)
    - [识别结果的基本结构](#%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C%E7%9A%84%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%84)
  - [域名](#%E5%9F%9F%E5%90%8D)
  - [接口](#%E6%8E%A5%E5%8F%A3)
- [接口详细](#%E6%8E%A5%E5%8F%A3%E8%AF%A6%E7%BB%86)
  - [图片识别](#%E5%9B%BE%E7%89%87%E8%AF%86%E5%88%AB)
  - [视频识别](#%E8%A7%86%E9%A2%91%E8%AF%86%E5%88%AB)
  - [查询视频识别结果](#%E6%9F%A5%E8%AF%A2%E8%A7%86%E9%A2%91%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C)
  - [场景参数和结果](#%E5%9C%BA%E6%99%AF%E5%8F%82%E6%95%B0%E5%92%8C%E7%BB%93%E6%9E%9C)
    - [剑皇](#%E5%89%91%E7%9A%87)
    - [鉴暴](#%E9%89%B4%E6%9A%B4)
    - [鉴政](#%E9%89%B4%E6%94%BF)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# API概览

## 提交任务参数说明

* 资源标识（URI）：支持HTTP/HTTPS、Base64 Data、Qiniu协议
* scene：识别场景，可执行对资源的识别场景，包括但不限于：鉴黄、鉴暴、鉴政等

## 识别结果说明

* suggestion：审核意见，取值范围```pass/block/review```
* label：识别的类别
* score：识别的置信度

### 识别结果的基本结构

```
{
    "code": 200,
    "message": "OK",
    "suggestion": "pass",
    "scenes": {
        "pulp": {
            "suggestion": "pass",
            "result": {}
        },
        ...
    }
}
```

## 域名

```
ai-censor.qiniuapi.com
```

## 接口

| API | 描述 |
|:--- |:--- |
| ```/v1/censor/image/recognition``` | 图片识别 |
| ```/v1/censor/image/feedback``` | 图片结果反馈 |
| ```/v1/censor/video/asyncrecognition``` | 视频识别 |
| ```/v1/censor/video/taskresults``` | 查询视频识别结果 |
| ```/v1/censor/video/feedback``` | 视频结果反馈 |

# 接口详细

## 图片识别

**Request**

```
POST /v1/censor/image/recognition HTTP/1.1
Content-Type: application/json

{
    "datas": [
        {
            "uri": "http://xxx.com/yy"
        },
        ...
    ]
    "scenes": ["pulp", "terror", "politicaion"],
    "params: {
        "scenes": {
            "pulp": {}
        }
    }
}
```

**Response**

```
200 OK
Content-Type: application/json

{
    "tasks": [
        {
            "code": 200,
            "message": "OK",
            "suggestion": "pass",
            "scenes": {
                "pulp": {
                    "suggestion": "pass",
                    "result": {
                        "label": "normal",
                        "score": 0.99
                    }
                },
                ...
            } 
        },
        ...
    ]
}
```

## 视频识别

**Request**

```
POST /v1/censor/video/asyncrecognition HTTP/1.1
Content-Type: application/json

{
    "datas": [
        {
            "uri": "http://xxx.com/yy"
        },
        ...
    ]
    "scenes": ["pulp", "terror", "politicaion"],
    "params: {
        "scenes": {
            "pulp": {}
        }
    }
}
```

**Response**

```
200 OK
Content-Type: application/json

{
    "tasks": [
        {
            "code": 200,
            "message": "OK",
            "job_id": <JobID>
        },
        ...
    ]
}
```

**Notify**

```
POST /xxx HTTP/1.1
Content-Type: application/json

{
    "code": 200,
    "message": "OK",
    "suggestion": "pass",
    "scenes": {
        "pulp": {
            "suggestion": "pass",
            "segments": [
                {
                    "offset_begin": <offset_begin>,
                    "offset_end": <offset_end>,
                    "suggestion": "pass",
                    "cuts": [
                        "offset": <offset>,
                        "uri": <uri>,
                        "suggestion": "pass",
                        "result": {
                           "label": "normal",
                            "score": 0.99 
                        }
                    ]
                },
                ...
            ]
        },
        ...
    } 
}
```

## 查询视频识别结果

**Request**

```
POST /v1/censor/video/taskresults HTTP/1.1
Content-Type: application/json

{
    "tasks": [
        {
            "task_id": <taskID>
        },
        ...
    ]
}
```

**Response**

```
200 OK
Content-Type: application/json

{
    "tasks": [
        {
            "task_id": <taskID>,
            "result": {
                "code": 200,
                "message": "OK",
                "suggestion": "pass",
                "scenes": {
                    "pulp": {
                        "suggestion": "pass",
                        "segments": [
                            {
                                "offset_begin": <offset_begin>,
                                "offset_end": <offset_end>,
                                "suggestion": "pass",
                                "cuts": [
                                    "offset": <offset>,
                                    "uri": <uri>,
                                    "suggestion": "pass",
                                    "result": {
                                       "label": "normal",
                                        "score": 0.99 
                                    }
                                ]
                            },
                            ...
                        ]
                    },
                    ...
                }
            }
        },
        ...
    ]
}
```

## 场景参数和结果

### 剑皇

**Scene**

```pulp```

**Params**

```nil```

**Result**

```
{
    "label": <label>,
    "score": <score>
}
```

| 字段 | 描述 |
| :--- | :--- |
| ```label``` | 类别，取值范围: ```pulp```,```sexy```,```normal``` |
| ```score``` | 类别准确度 |

### 鉴暴

**Scene**

```terror```

**Params**

```nil```

**Result**

```
{
    "label": <label>,
    "score": <score>
}
```

| 字段 | 描述 |
| :--- | :--- |
| ```label``` | 类别，取值范围: ```normal```,```terror```,其他详细分类 |
| ```score``` | 类别准确度 |

### 鉴政

**Scene**

```politician```

**Params**

```nil```

**Result**

```
{
    "label": <label>,
    "faces": [
        {
            "bounding_box": {
                "pts": [[],[],[],[]],
                "score": 0.99
            },
            "faces": [
                {
                    "name": "xx",
                    "score": 0.88
                }
            ]
        }
    ]
}
```

| 字段 | 描述 |
| :--- | :--- |
| ```label``` | 类别，取值范围: ```normal```,```face```,```politician``` |
| ```faces.bounding_box.pts``` | 人脸坐标框 |
| ```faces.bounding_box.score``` | 人脸准确度 |
| ```faces.faces.name``` |  识别到的涉政人员姓名 |
| ```faces.faces.score``` |  识别的准确度 |