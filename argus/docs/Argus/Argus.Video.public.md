<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [视频分析（直播、点播）](#%E8%A7%86%E9%A2%91%E5%88%86%E6%9E%90%E7%9B%B4%E6%92%AD%E7%82%B9%E6%92%AD)
  - [视频分析API](#%E8%A7%86%E9%A2%91%E5%88%86%E6%9E%90api)
    - [POST /v1/video](#post-v1video)
    - [GET /v1/jobs/video](#get-v1jobsvideo)
    - [GET /v1/jobs/video/status](#get-v1jobsvideostatus)
    - [CALLBACK /v1/video/op-result](#callback-v1videoop-result)
    - [CALLBACK /v1/video/result](#callback-v1videoresult)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 视频分析（直播、点播）

## 视频分析API

* 同步API

	| PATH | 调用方式 | 说明 |
	| :--- | :--- | :--- |
	| [POST /v1/video](#post-/v1/video) | HTTP | 提交同步视频分析请求 |

* 异步API

	| PATH | 调用方式 | 说明 |
	| :--- | :--- | :--- |
	| [POST /v1/video](#post-/v1/video) | HTTP | 提交异步视频分析请求 |
	| [GET /v1/jobs/video](#get-/v1/jobs/video) | HTTP | 获取视频分析任务的结果 |
	| [GET /v1/jobs/video?status=](#get-/v1/jobs/video/status) | HTTP | 获取特定状态的视频分析任务 | 
	| [CALLBACK /v1/video/op-result](#callback-/v1/video/op-result) | Callback | 单op处理结束后的结果回调 |
	| [CALLBACK /v1/video/result](#callback-/v1/video/result) | Callback | 所有op处理结束后的结果回调 |

### POST /v1/video

> 提交视频分析请求（同步／异步）

*Request*

```
POST /v1/video/<video_id>
Content-Type: application/json

{
	"data": {
		"uri": "http://xx"
	},
	"params": {
		"async": <async:bool>,
		"vframe": {
			"mode": <mode:int>,
			"interval": <interval:float>
		},
		"save": {
			"bucket": <bucket:string>,
			"zone": <zone:int>,
			"prefix": <prefix:string>
		},
		"hookURL": "http://yy.com/yyy"
	},
	"ops": [
		{
			"op": <op:string>,
			"hookURL": "http://yy.com/yyy",
			"hookURL_segment": "http://yy.com/yyy",
			"params": {
				"labels": [
					{
						"label": <label:string>,
						"select": <select:int>,
						"score": <score:float>
					},
					...
				],
				"terminate": {
					"mode": <mode:int>,
					"labels": {
						<label>: <max:int>
					}
				}
			}
		},
		... 
	]
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `video_id` | string | Y | 视频唯一标识，异步处理的返回结果中会带上该信息 |
| `data.uri` | string | Y | 视频流地址 |
| `params.async` | boolean | X | 是否异步处理，默认值为`False` |
| `params.vframe.mode` | int | X | 截帧处理逻辑，可选值为`[0, 1]`，默认值为`1` |
| `params.vframe.interval` | int | X | 截帧处理参数 |
| `params.save.bucket` | string | X | 保存截帧图片的Bucket |
| `params.save.zone` | int | X | 保存截帧图片的Zone |
| `params.save.prefix` | string | X | 截帧图片名称的前缀，图片名称的格式为`<prefix>/<video_id>/<offset>` （图片命名格式仅供参考，业务请不要依赖此命名格式）|
| `params.hookURL` | string | X | 所有op处理结束后的回调地址 |
| `ops.op` | string | Y | 执行的推理cmd |
| `ops.op.hookURL` | string | X | 该op处理结束后的回调地址 |
| `ops.op.hookURL_segment` | string | X | 片段回调地址 |
| `ops.op.params.labels.label` | string | X | 选择类别名，跟具体推理cmd有关 |
| `ops.op.params.labels.select` | int | X | 类别选择条件，`1`表示忽略不选；`2`表示只选该类别 |
| `ops.op.params.labels.score` | float | X | 类别选择的可信度参数，当select=`1`时表示忽略不选小于score的结果，当select=`2`时表示只选大于等于该值的结果 |
| `ops.op.params.terminate.mode` | int | X | 提前退出类型。`1`表示按帧计数；`2`表示按片段计数 |
| `ops.op.params.terminate.label.max` | int | X | 该类别的最大个数，达到该阈值则处理过程退出 |

> * `params.vframe`
> 	* `mode==0`: 每隔固定时间截一帧，由`vframe.interval`指定，单位`s`, 默认为`5.0s`
> 	* `mode==1`: 截关键帧, 默认为该模式

> | OP | Labels | Result |
> | :--- | :--- | :--- |
> | `pulp` | `0`, `1`, `2` | ARGUS - `/v1/pulp` |
> | `terror` | `0`, `1` | ARGUS - `/v1/terror` |
> | `terror_detect` | `__background__`,`isis flag`,`islamic flag`,`knives`,`guns`,`tibetan flag` | SERVING - `/v1/eval/terror-detect` |
> | `terror_classify` | `normal`,`bloodiness`,`bomb`,`march`,`beheaded`,`fight` | SERVING - `/v1/eval/terror-classify` |
> | `face_group_search` | 人名 | ARGUS - `/v1/face/group/<gid>/search` |
> | `detection` | 物体分类名称 | SERVING - `/v1/eval/detection` |
> | `politician` | 人名 | ARGUS - `/v1/face/search/politician` |
> | `image_label` | 图片打标 | ARGUS - `/v1/image/label` |

*Response*

*`When params.async == False`*

> 同步视频分析请求的返回结果

```
HTTP/1.1 200 OK
Content-Type: application/json

{
	<op:string>: {
		"labels": [
			{
				"label": <label:string>,
				"score": <score:float>
			},
			...
		],
		"segments": [
			{
				"offset_begin": <offset:int>,
				"offset_end": <offset:int>,
				"labels": [
					{
						"label": <label:string>,
						"score": <score:float>
					},
					...
				],
				"cuts": [
					{
						"offset": <offset:int>,
						"uri": <uri:string>,
						"result": {}
					},
					...
				]	
			},
			...
		]
	}
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `op` | string | Y | 推理cmd |
| `op.labels.label` | string | Y | 视频的判定结果 |
| `op.lables.score` | float | Y | 视频的判定结果可信度 |
| `op.segments.offset_begin` | int | Y | 片段起始的时间位置 |
| `op.segments.offset_end` | int | Y | 片段结束的时间位置 |
| `op.segments.labels.label` | string | Y | 片段的判定结果 |
| `op.segments.labels.score` | float | Y | 片段的判定结果可信度 |
| `op.segments.cuts.offset` | int | Y | 截帧的时间位置 |
| `op.segments.cuts.uri` | string | X | 截帧的保存路径 |
| `op.segments.cuts.result` | interface | Y | 截帧的对应的结果，内容跟具体推理相关 |

*`When params.async == True`*

> 异步视频分析请求的返回结果（立即返回）

```
HTTP/1.1 200 OK
Content-Type: application/json

{
	"job": <job_id:string>
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `job_id` | string | Y | 视频分析任务唯一标识 |

*错误信息*

| Error Code | Error Message | Desc |
| :--- | :--- | :--- |
| 400 | "bad op" | 输入的op值不支持 |
| 400 | "invalid interval, allow interval is [0, 10]" | 输入的vframe的interval值不在[0-10]的范围 |
| 400 | "invalid mode, allow mode is [0, 1]" | 输入的vframe的mode值不在[0-1]的范围 |
| 400 | "invalid starttime" | 输入的vframe的ss值小于0 |
| 400 | "invalid duration" | 输入的vframe的t值小于0 |
| 400 | "invalid parameters" | 输入的视频参数错误 |
| 424 | "cannot find the video" | 找不到输入的视频文件 |
| 400 | "cannot open the file" | 打不开输入的视频文件 |
| 500 | "cannot allow memory" | 发生内存不足等情况 |

### GET /v1/jobs/video

> 获取视频分析任务的结果

*Request*

```
GET /v1/jobs/video/<job_id> HTTP/1.1
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `job_id` | string | Y | 视频任务唯一标识，申请任务时返回的`job_id` |

*Response*

```
{
	"id": <string>,
	"vid": <string>,
	"request": {},
	"status": <string>,
	"result": {
		<op>: {
		},
		...
	},
	"error": <string>,
	"created_at": <string>,
	"updated_at": <string>
}
```

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | string | 视频任务唯一标识 |
| `vid` | string | 视频唯一标识，申请任务时传入的`id` |
| `request` | object | 视频分析请求 |
| `status` | string | 任务状态，`WAITING|DOING|FINISHED|` |
| `result` | object | 详细解释见 [CALLBACK /v1/video/op-result](#callback-/v1/video/op-result) |
| `error` | string | 处理视频的过程中遇到的错误，会返回相应的错误信息，详细解释见 [POST /v1/video](#post-/v1/video) |
| `created_at` | string | 任务创建时间，like: `2006-01-02T15:03:04` |
| `updated_at` | string | 任务更新时间，like: `2006-01-02T15:03:04` |


### GET /v1/jobs/video/status

> 获取特定状态的视频分析任务

*Request*

```
GET /v1/jobs/video?status=<string> HTTP/1.1

```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `status` | string | X | 任务状态，`WAITING|DOING|FINISHED` |

*Response*

```
[
	{
		"id": <string>,
		"status": <string>,
		"created_at": <string>,
		"updated_at": <string>
	},
	...
]
```

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | string | 任务唯一标识 |
| `status` | string | 任务状态，`WAITING|DOING|FINISHED` |
| `created_at` | string | 任务创建时间，like: `2006-01-02T15:03:04` |
| `updated_at` | string | 任务更新时间，like: `2006-01-02T15:03:04` |


### CALLBACK /v1/video/op-result

> 单op处理结束后的结果回调

*Request*

```
POST /xxxxx HTTP/1.1
Content-Type: application/json

{
	"id": <video_id:string>,
	"op": <op:string>,
	"result": {
		"labels": [
			{
				"label": <label:string>,
				"score": <score:float>
			},
			...
		],
		"segments": [
			{
				"offset_begin": <offset:int>,
				"offset_end": <offset:int>,
				"labels": [
					{
						"label": <label:string>,
						"score": <score:float>
					},
					...
				],
				"cuts": [
					{
						"offset": <offset:int>,
						"uri": <uri:string>,
						"result": {}
					},
					...
				]	
			},
			...
		]
	}
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `id` | string | Y | 视频唯一标识，申请任务时传入的`id` |
| `op` | string | Y | 推理cmd |
| `result.labels.label` | string | Y | 视频的判定结果 |
| `result.lables.score` | float | Y | 视频的判定结果可信度 |
| `result.segments.offset_begin` | int | Y | 片段起始的时间位置 |
| `result.segments.offset_end` | int | Y | 片段结束的时间位置 |
| `result.segments.labels.label` | string | Y | 片段的判定结果 |
| `result.segments.labels.score` | float | Y | 片段的判定结果可信度 |
| `result.segments.cuts.offset` | int | Y | 截帧的时间位置 |
| `result.segments.cuts.uri` | string | X | 截帧的保存路径 |
| `result.segments.cuts.result` | interface | Y | 截帧的对应的结果，内容跟具体推理相关 |


### CALLBACK /v1/video/result

> 所有op处理结束后的结果回调

*Request*

```
POST /xxxxx HTTP/1.1
Content-Type: application/json

{
	"id": <video_id:string>,
	"result": {
		<op>: {
		},
		...
	}
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `id` | string | Y | 视频唯一标识，申请任务时传入的`id` |
| `result` | object | Y | 详细解释见 [CALLBACK /v1/video/op-result](#callback-/v1/video/op-result) |


