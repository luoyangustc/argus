<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [API](#api)
  - [/face/group/new](#facegroupnew)
  - [/face/group/remove](#facegroupremove)
  - [/face/group/add](#facegroupadd)
  - [/face/group/delete](#facegroupdelete)
  - [/face/group](#facegroup)
  - [/face/group/<id>](#facegroupid)
  - [/face/group/search (已废弃)](#facegroupsearch-%E5%B7%B2%E5%BA%9F%E5%BC%83)
  - [/face/groups/search](#facegroupssearch)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

基本语义说明
* 资源表示方式（URI）。通过统一方式定位、获取资源（图片、二进制数据等）
	* HTTP， 网络资源，形如：`http://host/path`、`https://host/path`
	* Data，Data URI Scheme形态的二进制文件，形如：`data:application/octet-stream;base64,xxx`。
	
# API

| PATH | Note | Input | Response Type |
| :--- | :--- | :--- | :---: |
| [`/v1/face/group/<id>/new`](#/face/group/new) | 新建人脸库 | image URI | Json |
| [`/v1/face/group/<id>/remove`](#/face/group/remove) | 删除人脸库 | Json | Json |
| [`/v1/face/group/<id>/add`](#/face/group/add) | 人脸库添加特征 | image URI | Json |
| [`/v1/face/group/<id>/delete`](#/face/group/delete) | 人脸库删除特征 | Json | Json |
| [`/v1/face/group`](#/face/group) | 显示所有人脸库 | GET | Json |
| [`/v1/face/group/<id>`](#/face/group/<id>) | 显示人脸库特征 | GET | Json |
| [`/v1/face/group/<id>/search`](#/face/group/search) | 人脸库搜索 | image URI | Json |

## /face/group/new

> 新建人脸库

Request

```
POST /v1/face/group/<id>/new  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"name": <name>,
				"mode": <mode>,
				"desc": <additional information>
			}
		},
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"name": <name>,
				"mode": <mode>,
				"desc": <additional information>
			}
		}
	]
}
```

Response

```
200 OK
Content-Type: application/json

{
	"faces": [
		<face_id>,
		"",
		...
	],
	"attributes": [
		{
			"bounding_box": {
				"pts": [[205,400],[1213,400],[1213,535],[205,535]],
				"score":0.998
			}
		},
		null,
		...
	],
	"errors": [
		null,
		{
			"code": xxx,
			"message": "xxx"
		},
		...
	]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|
|uri|string|图片资源地址，其中包含且只包含一张人脸|
|id|string|人脸ID，可选|
|name|string|人物姓名|
|mode|string|人脸选择策略，可以设置为 SINGLE（只允许图片里面出现单张人脸，否则API返回错误） 或者 LARGEST（如果存在多张人脸，使用最大的人脸），不填默认 SINGLE，人脸不能小于50*50 |
|desc|map|人脸图片的备注信息，可选。内容为json，最大允许长度为4096字节 |

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|faces|list|录入的所有人脸id列表，若某张人脸入库失败，则为空|
|attributes|list|录入的所有人脸相关信息，若某张人脸入库失败，则为null|
|bounding_box|map|图片中某张人脸的边框信息|
|bounding_box.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|bounding_box.score|float|人脸位置检测准确度|
|errors|map|录入的所有人脸的错误信息，若某张人脸入库成功，则为null|
|code|int|错误码|
|message|string|错误描述信息|


## /face/group/remove

> 删除人脸库

Request

```
POST /v1/face/group/<id>/remove  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 ok

```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|


## /face/group/add

> 人脸库新增特征

Request

```
POST /v1/face/group/<id>/add  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"name": <name>,
				"mode": <mode>,
				"desc": <additional information>
			}
		},
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"name": <name>,
				"mode": <mode>,
				"desc": <additional information>
			}
		}
	]
}
```

Response

```
200 OK
Content-Type: application/json

{
	"faces": [
		<face_id>,
		"",
		...
	],
	"attributes": [
		{
			"bounding_box": {
				"pts": [[205,400],[1213,400],[1213,535],[205,535]],
				"score":0.998
			}
		},
		null,
		...
	],
	"errors": [
		null,
		{
			"code": xxx,
			"message": "xxx"
		},
		...
	]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|
|uri|string|图片资源地址，其中包含且只包含一张人脸|
|id|string|人脸ID，可选|
|name|string|人物姓名|
|mode|string|人脸选择策略，可以设置为 SINGLE（只允许图片里面出现单张人脸，否则API返回错误） 或者 LARGEST（如果存在多张人脸，使用最大的人脸），不填默认 SINGLE，人脸不能小于50*50 |
|desc|map|人脸图片的备注信息，可选。内容为json，最大允许长度为4096字节 |

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|faces|list|录入的所有人脸id列表，若某张人脸入库失败，则为空|
|attributes|list|录入的所有人脸相关信息，若某张人脸入库失败，则为null|
|bounding_box|map|图片中某张人脸的边框信息|
|bounding_box.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|bounding_box.score|float|人脸位置检测准确度|
|errors|map|录入的所有人脸的错误信息，若某张人脸入库成功，则为null|
|code|int|错误码|
|message|string|错误描述信息|


## /face/group/delete

> 删除人脸库特征

Request

```
POST /v1/face/group/<id>/delete  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"faces":[
		<face_id>,
		...
	]
}

```

Response

```
200 ok

{
	"code": 0,
	"message": "",
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|
|face_id|string|人脸唯一标识|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|


## /face/group

> 显示所有的人脸库

Request

```
GET /v1/face/group  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": [
		<id>,
		...
	]
}
```
返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|id|string|人脸库唯一标识|

## /face/group/<id>

> 显示人脸库

Request

```
GET /v1/face/group/<id>  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": [
		{
			"id": <id>,
			"value": {
				"name": "xx"
			}
		},
		...
	]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|id|string|生成的该人脸的唯一标识|
|value.name|string|人物姓名|

## /face/group/search (已废弃)

> 人脸库搜索。对于待搜索图片中检测到的每张人脸，返回其相似度最高的人脸

Request

```
POST /v1/face/group/<id>/search Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://xx.com/xxx"
	}
}
```

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"detections": [
			{
				"boundingBox": {
					"pts": [[205,400],[1213,400],[1213,535],[205,535]],
					"score":0.998
				},
				"value": {
					"boundingBox": {
						"pts": [[105,200],[678,200],[678,480],[105,480]],
						"score":0.956
					},
					"id": "xx",
					"name": "xx",
					"score":0.9998,
					"desc": <additional information>
				}
			}
		]
	}
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|图片库的唯一标识|
|data.uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result.detections|list|图片中检测到的所有人脸|
|result.detections.[].boundingBox|map|图片中某张人脸的边框信息|
|result.detections.[].value|map|对于图片中某张人脸，检索出的最相似的人脸信息|
|result.detections.[].value.boundingBox|map|对于图片中某张人脸，检索出的最相似的人脸坐标|
|boundingBox.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|boundingBox.score|float|人脸位置检测准确度|
|id|string|检索得到的人脸唯一标识|
|name|string|检索得到的人物姓名|
|score|float|0~1,检索结果的可信度，1为确定|
|desc|map|人脸图片入库时的备注信息|


## /face/groups/search

> 人脸库搜索。对于待搜索图片中检测到的每张人脸，在指定的人脸库中返回其相似度最高的多张人脸

Request

```
POST /v1/face/groups/search Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://xx.com/xxx"
	},
	"params": {
		"groups": [
			<group_id>
		],
		"limit": 5,
		"threshold": 0.85
	}
}
```

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"faces": [
			{
				"bounding_box": {
					"pts": [[205,400],[1213,400],[1213,535],[205,535]],
					"score":0.998
				},
				"faces": [
					{
						"bounding_box": {
							"pts": [[105,200],[678,200],[678,480],[105,480]],
							"score":0.956
						},
						"id": "xx",
						"name": "xx",
						"group": "xx",
						"score":0.9998,
						"desc": <additional information>
					},
					...
				]
			}
		]
	}
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|data.uri|string|图片资源地址|
|params.groups|list|搜索的人脸库列表，必选，最多5个|
|params.groups.[]|string|人脸库id|
|params.limit|int|匹配人脸TOPN，可选，默认为1，最大允许10000。若为-1，则返回所有匹配人脸|
|params.threshold|float|匹配人脸的精度阈值，可选，默认使用系统设置值。若该值小于系统设置值，则仍使用系统设置值|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result.faces|list|图片中检测到的所有人脸|
|result.faces[].bounding_box|map|图片中某张人脸的边框信息|
|result.faces.[].faces|list|对于图片中某张人脸，检索出的相似人脸列表|
|result.faces[].faces[].bounding_box|map|检索得到的人脸的边框信息|
|bounding_box.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|bounding_box.score|float|人脸位置检测准确度|
|id|string|检索得到的人脸唯一标识|
|name|string|检索得到的人物姓名|
|group|string|检索得到的人脸所属的人脸库id|
|score|float|0~1,检索结果的可信度，1为确定|
|desc|map|人脸图片入库时的备注信息|