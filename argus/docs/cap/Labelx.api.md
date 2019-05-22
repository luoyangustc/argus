<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [资源实体](#%E8%B5%84%E6%BA%90%E5%AE%9E%E4%BD%93)
  - [GetAuditorAttrResp](#getauditorattrresp)
  - [LabelInfo](#labelinfo)
  - [TaskResult](#taskresult)
  - [GetRealtimeTaskResp](#getrealtimetaskresp)
  - [PostResultReq](#postresultreq)
- [API](#api)
  - [GET /v1/audit/auditor/attr/_](#getv1auditauditorattr_)
  - [GET /v1/audit/realtime/task/_](#getv1auditrealtimetask_)
  - [POST /v1/audit/cancel/realtime/task](#post-v1auditcancelrealtimetask)
  - [POST /v1/audit/result](#post-v1auditresult)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# 资源实体
## GetAuditorAttrResp
>  Auditor的信息

```
<GetAuditorAttrResp.object>
{
    auditorId:<string>,
    realTimeLevel:<string>,
    curLabel:<string>,
    valid:<string>,
}
```

| FIELD | NOTE | RANGE | DEFAULT | MANDATORY |
| :--- | :--- | :--- | :--- | :--- |
| auditorId |标注人员的id | | |  |
| realTimeLevel |实时或非实时 |"batch", "realtime" |"batch" |  |
| curLabel | 可标注任务类型 |  | |  |
| valid | Auditor当前是否处于正常工作状态 |"valid", "invalid"  |"invalid" |  |

## LabelInfo

```
<LabelInfo.object>
{	
    name:<string>,
    type:<string>,
    data:<[]interface{}>,
}
```
### 参数说明
当name为pulp或terror时，data的结构为：
```
[]struct{
	class:<string>,
	score:<string>,
}
```

当name为politician时，data的结构为:
```
[]struct{
    class:<string>,
    faces:[]struct{
        bounding_box: struct{
            pts:<[][2]int>,
            score:<float32>
        },
        faces: []struct{
            id:<string>,
            name:<string>,
            score:<float32>,
            group:<string>,
            sample<*struct{
                url:<string>,
                pts:<[][2]int>
            }>
        }
    }
}
```

| FIELD | NOTE | RANGE | DEFAULT | MANDATORY |
| :--- | :--- | :--- | :--- | :--- |
| name |操作类型:pulp,terror,... | | |  |
| type | classification， detection，有政治人物的时候为detection，否则为classification|  | |  |
| data.class | 图片所属的类型：sexy/xxx/.../bikini, 粗/.../细 |  | |  |
| data.score | 图片所属的类型的得分，人审的该参数忽略 |  | |  |
| data.faces.bounding_box.pts | 空间区域 |  | |  |
| data.faces.bounding_box.score | 针对细标签的置信度 |  | |  |
| data.faces.faces.id| |  | |  |
| data.faces.faces.name | 政治人物的姓名 |  | |  |
| data.faces.faces.score | 该区域框中的审核结果的置信度，人审忽略该参数 |  | |  |
| data.faces.faces.group | 政治人物所属的分组 |  | |  |
| data.faces.faces.sample.url | 底库的url |  | |  |
| data.faces.faces.sample.pts | 底库的空间区域 |  | |  |

## TaskResult 

```
<TaskResult.object>
{	
    taskId:<string>,
    url:<string>,
    label:<[]LabelInfo.object>,
}
```

| FIELD | NOTE | RANGE | DEFAULT | MANDATORY |
| :--- | :--- | :--- | :--- | :--- |
| taskId |Task ID | | |  |
| url | 图片地址 |  | |  |
| label | 图片的结果，详见 #labelinfo |  | |  |

## GetRealtimeTaskResp
> 请求实时任务返回给labelx的结果格式
```
<GetRealtimeTaskResp.object>
{
	auditorId:<string>,
	type:<string>,
	taskType:<[]string>,
	labels:<map[string][]LabelTitle.object>,
	callbackUri:<string>,
	mode:<string>, 
	indexData:<[]TaskResult.object>,
    expiryTime:<string>,
}
```

```
<LabelTitle.object>
{
    title:<string>,
    desc:<string>,
    selected:<bool>,
}
```


| FIELD | NOTE | RANGE | DEFAULT | MANDATORY |
| :--- | :--- | :--- | :--- | :--- |
| auditorId |Auditor的ID | | |  |
| type | Image or Video |  | |  |
| taskType | 分类，检测 | 如果有politician，就都改成 detect.xxx，例如detect.politician; 没有就保持classify.xxx,如classify.pulp, classify.terror | |  |
| labels.title | 结果类型，例如normal，sexy。。。 |  | |  |
| labels.desc | 结果类型的中文描述，例如'正常',。。。 |  | |  |
| labels.selected | 是否选中该结果类型 |  | |  |
| callbackUri | 数据集标注数据回调地址 |  | |  |
| mode | 实时，非实时 | "batch", "realtime" | |  |
| indexData | 源数据集索引文件,要审核的图片及其机审结果，详见 (#taskresult) |  | |  |
| expiryTime | 超时时间，精确到秒 |  | |  |

## PostResultReq
> Labelx完成返回给人审的结果格式

```
<PostResultReq.object>
{
    auditorId:<string>,
    pid:<string>,
    success:<bool>,
    result:<[]TaskResult.object>
}
```

| FIELD | NOTE | RANGE | DEFAULT | MANDATORY |
| :--- | :--- | :--- | :--- | :--- |
| auditorId |Auditor的ID | | |  |
| pid | package id |  | |  |
| success | 该package里面的任务是否成功 |  | |  |
| result | 所有图片的审核结果，详见#taskresult |  | |  |


# API
## GET/v1/audit/auditor/attr/_

> 获取Auditor的基本信息，例如支持标注哪些业务类型等

### 请求

```
GET /v1/audit/auditor/attr/{auditorID}
```

### 返回（#getAuditorAttrResp）

```
200 OK
Content-Type: application/json
{
    "auditorId": "aid",
    "realTimeLevel": "realtime",
    "curLabel": "pulp",
    "valid": "invalid",
}
```

### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `auditorId` | string | Auditor的ID |
| `realTimeLevel` | byte | Auditor是实时或非实时 |
| `labelType` | []string | Auditor可标注任务类型 |
| `valid` | bool | Auditor当前是否处于正常工作状态 |

## GET/v1/audit/realtime/task/_

> 获取实时任务

### 请求

```
Get /v1/audit/realtime/task/{auditorID}
```

### 返回 (#getRealtimeTaskResp)

```
200 OK

{
    "auditorId":"AuditorId",
    "pid":"pid",
    "type":"Image",
    "taskType":"",
    "label":<label>,
    "callbackUri":"",
    "mode":"",
    "indexData":[],
    "expiryTime":""
}
```

### 参数说明
> 详见 #getRealtimeTaskResp

## POST /v1/audit/cancel/realtime/task
> Labelx取消当前标注任务

### 请求

```
POST /v1/audit/cancel/realtime/task

{
    "auditorId": "test",
    "pid":"pid",
    "taskIds":["taskId1", "taskId2"]
}
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `auditorId` | string | Y | Auditor的ID |
| `pid` | string | Y | Auditor的ID |
| `taskIds` | []string | Y | 取消的标注任务ID数组 |

### 返回

```
{
}
```

## POST /v1/audit/result
> Auditor完成标注工作，返回结果

### 请求 (#postresultreq)

```
POST /v1/audit/result
     
{
    "auditorId":"auditorId",
    "pid":"pid",
    "success":true,
    "result":[

    ]
}
```

### 参数说明 （详见 #postresultreq）

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `auditorId` | string | Y | Auditor的ID |
| `pid` | string | Y | 发给auditor的package id |
| `success` | bool | Y | true成功，false撤销 |
| `result` |  TaskResult.object | Y | 详见 #taskresult |

### 返回

```
{

}
```