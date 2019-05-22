<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1. 布控任务](#1-%E5%B8%83%E6%8E%A7%E4%BB%BB%E5%8A%A1)
  - [1.1 新建布控任务](#11-%E6%96%B0%E5%BB%BA%E5%B8%83%E6%8E%A7%E4%BB%BB%E5%8A%A1)
    - [1.1.1 人脸1:N搜索](#111-%E4%BA%BA%E8%84%B81n%E6%90%9C%E7%B4%A2)
  - [1.2 结束布控任务](#12-%E7%BB%93%E6%9D%9F%E5%B8%83%E6%8E%A7%E4%BB%BB%E5%8A%A1)
  - [1.3 查询所有布控任务](#13-%E6%9F%A5%E8%AF%A2%E6%89%80%E6%9C%89%E5%B8%83%E6%8E%A7%E4%BB%BB%E5%8A%A1)
  - [1.4 查询布控任务详情](#14-%E6%9F%A5%E8%AF%A2%E5%B8%83%E6%8E%A7%E4%BB%BB%E5%8A%A1%E8%AF%A6%E6%83%85)
- [2. 推理结果](#2-%E6%8E%A8%E7%90%86%E7%BB%93%E6%9E%9C)
  - [2.1 帧回调](#21-%E5%B8%A7%E5%9B%9E%E8%B0%83)
    - [1.2.1 人脸1：N搜索回调](#121-%E4%BA%BA%E8%84%B81%EF%BC%9An%E6%90%9C%E7%B4%A2%E5%9B%9E%E8%B0%83)
- [3. 任务结果](#3-%E4%BB%BB%E5%8A%A1%E7%BB%93%E6%9E%9C)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# 1. 布控任务

## 1.1 新建布控任务

> 新建对某路视频流的布控任务，指定布控内容和相关参数

**Request**
```
POST /v1/live/<live_id> HTTP/1.1
Context-Type: application/json

{
	"data": {
		"uri": "rtsp://xx"
	},
	"params": {
		"vframe": {
			"interval": <interval:float>
		},
		"live": {
			"timeout": <timeout:float>
		},
		"hookURL": "http://yy.com/yyy",
		"save": {
			"prefix": <save_path_prefix>
		}
	},
	"ops": [
		{
			"op": <op:string>,
			"cut_hook_url": "http://yy.com/yyy",
			"params": {
				...
			}
		},
		... 
	]
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `live_id` | string | Y | 视频唯一标识，异步处理的返回结果中会带上该信息 |
| `data.uri` | string | Y | 视频流地址 |
| `params.vframe.interval` | float | X | 截帧间隔参数，单位sec |
| `params.live.timeout` | float | X | 视频流结束判定参数，单位sec |
| `params.hookURL` | string | X | 所有op处理结束后的回调地址 |
| `params.save` | {}struct | X | 是否开启截帧保存，保存回调结果对应图片 |
| `params.save.prefix` | string | X | 截帧保存路径前缀 |
| `ops.op` | string | Y | 执行的推理cmd |
| `ops.op.hookURL_cut` | string | X | 截帧回调地址 |
| `ops.op.params` | {}struct | X | 类别参数，跟具体推理cmd有关 |

**Response**
```
200 OK
Content-Type: application/json

{
	"job": <job_id:string>
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `job` | string | Y | 视频布控任务唯一标识 |

### 1.1.1 人脸1:N搜索

> OP: face_group_search_private

```
{
	...
	"ops": [
		{
			"op": "face_group_search_private",
			"cut_hook_url": "http://yy.com/yyy",
			"params": {
				"other": {
					"group": <group_id>,
					"threshold": <search_threshold>,
					"limit": <search_result_limit>
				} 
			}
		},
		...
	]
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `ops.op` | string | Y | 执行的推理cmd |
| `ops.op.hookURL_cut` | string | X | 截帧回调地址 |
| `ops.op.params.other.group` | string | Y | 人脸搜索库ID，唯一标识人脸库 |
| `ops.op.params.other.threshold` | float | N | 人脸搜索相似度阈值，即相似度高于该值才返回结果，默认为0 |
| `ops.op.params.other.limit` | int | N | 相似人脸项数目限制，默认返回全部相似人脸 |

## 1.2 结束布控任务

> 结束对某路视频流的布控任务

**Request**
```
POST /v1/jobs/<job_id>/kill HTTP/1.1

```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `job_id` | string | Y | 布控任务的唯一标识，新建布控任务时返回的结果 |

**Response**
```
200 OK
```

## 1.3 查询所有布控任务

> 查询所有布控任务

**Request**

```
GET /v1/jobs/live?status=<job_status>&created_from=<created_from>&created_to=<created_to>&marker=<marker>&limit=<limit> HTTP/1.1
```

| 参数 | 类型 |  必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `status` | string| N | 指定查询任务状态,[""],默认为空，即查询所有任务 |
|`created_from`|string|视频布控任务查询的创建时间区间起点，格式为unix时间戳, 时间区间：[created_from,created_to)|
|`created_to`|string|视频布控任务查询的创建时间区间终点，格式为unix时间戳, 时间区间：[created_from,created_to)|
| `marker` | string | N | 上一次列举返回的位置标记，作为本次列举的起点信息。默认值为空字符串 |
| `limit` | string | 本次列举的条目数，范围为 1-1000。默认值为 1000。 |


**Response**
```
200 OK
Content-Type: application/json

{
	"jobs": [
		{
			"id": <job_id>,
			"live": <live_id>,
			"status": <job_status>,
			"created_at": <create_timestamp>
			"updated_at": <updapte_timestamp>
		}
		...
	],
	"marker":<Marker>
}
```

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `jobs.id` | string | 视频布控任务唯一标识 |
| `jobs.live` | string | 视频唯一标识，异步处理的返回结果中会带上该信息 |
| `jobs.status` | string | 视频布控任务状态 |
| `jobs.created_at` | string | 视频布控任务创建时间 |
| `jobs.updated_at` | string | 视频布控任务最近更新时间 |
| `marker` | string | 有剩余条目则返回非空字符串，作为下一次列举的参数传入。如果没有剩余条目则返回空字符串。 |


## 1.4 查询布控任务详情
> 根据job_id查询布控任务详情

**Request**

```
GET /v1/jobs/live/<job_id>
```

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `jobs_id` | string | 视频布控任务唯一标识 |

**Response**
```
200 OK
Content-Type: application/json

{
	"id": <job_id>
	"vid": <live_id>,
	"request": {

		"data": {
			"uri": "rtsp://xx"
		},
		"params": {
			"vframe": {
				"interval": <interval:float>
			},
			"live": {
				"timeout": <timeout:float>
			},
			"hookURL": "http://yy.com/yyy",
			"save": {
				"prefix": <save_path_prefix>
			}
		},
		"ops": [
			{
				"op": <op:string>,
				"cut_hook_url": "http://yy.com/yyy",
				"params": {
					...
				}
			},
			... 
		]
	},
	"status": <job_status>,
	"created_at": <create_timestamp>,
	"updated_at": <updapte_timestamp>,
	"error": <error_message>
}
```

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | string | 视频布控任务唯一标识 |
| `vid` | string | 视频唯一标识，异步处理的返回结果中会带上该信息 |
| `request.data.uri` | string | 视频流地址 |
| `request.params.vframe.interval` | float | 截帧间隔参数，单位sec |
| `request.params.live.timeout` | float | 视频流结束判定参数，单位sec |
| `request.params.hookURL` | string | 所有op处理结束后的回调地址 |
| `request.params.save` | {}struct | 是否开启截帧保存，保存回调结果对应图片 |
| `request.params.save.prefix` | string | 截帧保存路径前缀 |
| `request.ops.op` | string | 执行的推理cmd |
| `request.ops.op.hookURL_cut` | string | 截帧回调地址 |
| `request.ops.op.params` | {}struct | 类别参数，跟具体推理cmd有关 |
| `status` | string | 视频布控任务状态 |
| `created_at` | string | 视频布控任务创建时间 |
| `updated_at` | string | 视频布控任务最近更新时间 |
| `error` | string | 处理视频流的过程中遇到的错误，会返回相应的错误信息|


# 2. 推理结果

## 2.1 帧回调

**Requeset**
```
POST /xxxxxxx HTTP/1.1
Content-Type: application/json

{
	"live_id": <live_id:string>,
	"job_id": <job_id:string>
	"op": <op:string>,
	"offset": <offset:int>,
	"uri": <uri:string>,
	"result": {}
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `live_id` | string | Y | 视频唯一标识，申请任务时传入的`live_id` |
| `job_id` | string | 布控任务唯一标识 |
| `op` | string | Y | 推理cmd |
| `offset` | int | Y | 截帧时间 |
| `uri` | string | X | 截帧的保存路径 |
| `result.result` | interface | Y | 截帧的对应的结果，内容跟具体推理相关 |


**Response**
```
200 OK
```

### 1.2.1 人脸1：N搜索回调

> OP: face_group_search_private

```
{
	...
	"result": {
		"faces": [
			{
				"bounding_box": {
					"pts": [[606,515],[649,515],[649,576],[606,576]],
					"score":0.9916495					
				},
				"faces":[
					{
						"id":"tkLwryRJ0vUT2-ZYZO-dCQ==",
						"score":0.6523004,
						"tag":"张三",
						"desc":""
					}
					...
				]	
			}
			...
		]
	}
	...
}
```
| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| faces.bounding_box.pts     | array | 人脸所在图片中的位置，四点坐标值 |[左上，右上，右下，左下] 四点坐标框定的脸部 |
| faces.[].bounding_box.score     | float | 人脸的检测置信度 |
| faces.faces     | object | 和此人脸相似的人脸的列表 |
| faces.faces.id     | string | 相似人脸ID |
| faces.faces.score     | float | 两个人脸的相似度 |
| faces.faces.tag     | string | 相似人脸Tag |
| faces.faces.desc     | string | 相似人脸描述 |

# 3. 任务结果

**Requeset**
```
POST /xxxxxxx HTTP/1.1
Content-Type: application/json

{
	"live_id": <live_id:string>,
	"job_id": <job_id:string>,
	"code": <code:int>,
	"message": <message:string>
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `live_id` | string | Y | 视频唯一标识，申请任务时传入的`live_id` |
| `job_id` | string | 任务唯一标识 |
| `code` | int | Y | 布控任务错误码 |
| `message` | string | Y | 布控任务错误详细信息 |
