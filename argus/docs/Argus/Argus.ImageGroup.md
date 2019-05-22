<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [API](#api)
  - [/image/group/new](#imagegroupnew)
  - [/image/group/remove](#imagegroupremove)
  - [/image/group/add](#imagegroupadd)
  - [/image/group/delete](#imagegroupdelete)
  - [/image/group](#imagegroup)
  - [/image/group/<id>](#imagegroupid)
  - [/image/group/search](#imagegroupsearch)
  - [/image/groups/search](#imagegroupssearch)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

基本语义说明
* 资源表示方式（URI）。通过统一方式定位、获取资源（图片、二进制数据等）
	* HTTP， 网络资源，形如：`http://host/path`、`https://host/path`
	* Data，Data URI Scheme形态的二进制文件，形如：`data:application/octet-stream;base64,xxx`。
	
# API

| PATH | Note | Input | Response Type |
| :--- | :--- | :--- | :---: |
| [`/v1/image/group/<id>/new`](#/image/group/new) | 新建图片库 | image URI | Json |
| [`/v1/image/group/<id>/remove`](#/image/group/remove) | 删除图片库 | Json | Json |
| [`/v1/image/group/<id>/add`](#/image/group/add) | 图片库添加特征 | image URI | Json |
| [`/v1/image/group/<id>/del`](#/image/group/del) | 图片库删除特征 | Json | Json |
| [`/v1/image/group`](#/image/group) | 显示所有图片库 | GET | Json |
| [`/v1/image/group/<id>`](#/image/group/<id>) | 显示图片库特征 | GET | Json |
| [`/v1/image/group/<id>/search`](#/image/group/search) | 图片库搜索 | image URI | Json |


## /image/group/new

> 新建图片库

Request

```
POST /v1/image/group/<id>/new  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"label": <label>,
				"desc": <additional information>
			}
		},
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"label": <label>,
				"desc": <additional information>
			}
		}
	]
}
```

Response

```
200 ok

{
	"images": [
		<image_id>,
		"",
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
|id|string|图片库的唯一标识|
|uri|string|图片资源地址|
|attribute.id|string|图片唯一标识，可选|
|label|string|图片标识|
|desc|map|图片的备注信息，可选。内容为json，最大允许长度为4096字节 |

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|images|list|录入的所有图片id列表，若某张图片入库失败，则为空|
|errors|map|录入的所有图片的错误信息，若某张图片入库成功，则为null|
|code|int|错误码|
|message|string|错误描述信息|


## /image/group/remove

> 删除图片库

Request

```
POST /v1/image/group/<id>/remove  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 ok

```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|图片库的唯一标识|


## /image/group/add

> 图片库新增特征

Request

```
POST /v1/image/group/<id>/add  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"label": <label>,
				"desc": <additional information>
			}
		},
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"id": <id>,
				"label": <label>,
				"desc": <additional information>
			}
		}
	]
}
```

Response

```
200 ok

{
	"images": [
		<image_id>,
		"",
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
|id|string|图片库的唯一标识|
|uri|string|图片资源地址|
|attribute.id|string|图片唯一标识，可选|
|label|string|图片标识|
|desc|map|图片的备注信息，可选。内容为json，最大允许长度为4096字节 |

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|images|list|录入的所有图片id列表，若某张图片入库失败，则为空|
|errors|map|录入的所有图片的错误信息，若某张图片入库成功，则为null|
|code|int|错误码|
|message|string|错误描述信息|


## /image/group/delete

> 删除图片库特征

Request

```
POST /v1/image/group/<id>/delete  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"images":[
		<image_id>,
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
|id|string|图片库的唯一标识|
|image_id|string|图片唯一标识|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|


## /image/group

> 显示所有的图片库

Request

```
GET /v1/image/group  Http/1.1
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
|id|string|图片库唯一标识|

## /image/group/<id>

> 显示图片库

Request

```
GET /v1/image/group/<id>  Http/1.1
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
				"label": "xx"
			}
		},
		...
	]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|图片库的唯一标识|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|id|string|生成的该图片的唯一标识|
|value.label|string|图片标识|

## /image/group/search

> 图片库搜索。返回与待搜索图片最相似的多张图片

Request

```
POST /v1/image/group/<id>/search Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://xx.com/xxx"
	},
	"params": {
		"limit": 1
	}
}
```

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": [
		{
			"id": "xx",
			"label": "xx",
			"score":0.9998
			"uri": "http://xxx/xxx",
			"desc": <additional information>
		},
		...
	]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|图片库的唯一标识|
|uri|string|图片资源地址|
|params.limit|int|匹配图片TOPN，可选，默认为1，最大允许10000。若为-1，则返回所有匹配图片|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result|list|检索得到的图片列表|
|id|string|检索得到的图片id|
|uri|string|检索得到的图片地址|
|label|string|检索得到的图片标识|
|score|float|0~1,检索结果的可信度，1为确定|
|desc|map|图片入库时的备注信息|


## /image/groups/search

> 图片库搜索。在指定的图片库中，返回与待搜索图片最相似的多张图片

Request

```
POST /v1/image/groups/search Http/1.1
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
	"result": [
		{
			"id": "xx",
			"label": "xx",
			"score":0.9998
			"uri": "http://xxx/xxx",
			"group": "xx",
			"desc": <additional information>
		},
		...
	]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|params.groups|list|搜索的图片库列表，必选，最多5个|
|params.groups.[]|string|图片库id|
|params.limit|int|匹配图片TOPN，可选，默认为1，最大允许10000。若为-1，则返回所有匹配图片|
|params.threshold|float|匹配图片的精度阈值，可选，默认使用系统设置值。若该值小于系统设置值，则仍使用系统设置值|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result|list|检索得到的图片列表|
|id|string|检索得到的图片id|
|uri|string|检索得到的图片地址|
|label|string|检索得到的图片标识|
|score|float|0~1,检索结果的可信度，1为确定|
|group|string|检索得到的图片所属的图片库id|
|desc|map|图片入库时的备注信息|