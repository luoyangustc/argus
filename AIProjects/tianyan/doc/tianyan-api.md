<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1. 人脸库API](#1-%E4%BA%BA%E8%84%B8%E5%BA%93api)
  - [1.1.创建人脸库](#11%E5%88%9B%E5%BB%BA%E4%BA%BA%E8%84%B8%E5%BA%93)
  - [1.2. 删除人脸库](#12-%E5%88%A0%E9%99%A4%E4%BA%BA%E8%84%B8%E5%BA%93)
  - [1.3. 获取所有人脸库](#13-%E8%8E%B7%E5%8F%96%E6%89%80%E6%9C%89%E4%BA%BA%E8%84%B8%E5%BA%93)
  - [1.4. 添加人脸](#14-%E6%B7%BB%E5%8A%A0%E4%BA%BA%E8%84%B8)
  - [1.5. 删除人脸](#15-%E5%88%A0%E9%99%A4%E4%BA%BA%E8%84%B8)
  - [1.6. 人脸搜索](#16-%E4%BA%BA%E8%84%B8%E6%90%9C%E7%B4%A2)
- [2. 人脸聚类排重API](#2-%E4%BA%BA%E8%84%B8%E8%81%9A%E7%B1%BB%E6%8E%92%E9%87%8Dapi)
  - [2.1. 创建人脸聚类库](#21-%E5%88%9B%E5%BB%BA%E4%BA%BA%E8%84%B8%E8%81%9A%E7%B1%BB%E5%BA%93)
  - [2.2. 删除人脸聚类库](#22-%E5%88%A0%E9%99%A4%E4%BA%BA%E8%84%B8%E8%81%9A%E7%B1%BB%E5%BA%93)
  - [2.3. 人脸聚类搜索](#23-%E4%BA%BA%E8%84%B8%E8%81%9A%E7%B1%BB%E6%90%9C%E7%B4%A2)
- [3. 视频流处理服务API](#3-%E8%A7%86%E9%A2%91%E6%B5%81%E5%A4%84%E7%90%86%E6%9C%8D%E5%8A%A1api)
  - [3.1. 启动视频流处理服务](#31-%E5%90%AF%E5%8A%A8%E8%A7%86%E9%A2%91%E6%B5%81%E5%A4%84%E7%90%86%E6%9C%8D%E5%8A%A1)
  - [3.2. 停止视频流处理服务](#32-%E5%81%9C%E6%AD%A2%E8%A7%86%E9%A2%91%E6%B5%81%E5%A4%84%E7%90%86%E6%9C%8D%E5%8A%A1)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1. 人脸库API

## 1.1.创建人脸库
> 新建人脸库 /face/group/new

**Request**

```
 POST /v1/face/group/<id>/new
 Content-Type: application/json
 {
     "size": 100000,
     "precision": 4,
     "dimension": 512
 }
```

**Respose**

```
200 ok
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|
|size|int|人脸库预期特征数目，必填|
|precision|int|特征精度，选填，默认precision=4，即float32|
|dimension|int|特征维度，选填，默认dimension=512|

## 1.2. 删除人脸库
> 删除人脸库 /face/group/remove

**Request**

```
 POST /v1/face/group/<id>/remove

```

**Respose**

```
200 ok
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|

## 1.3. 获取所有人脸库
> 获取所有人脸信息 /face/group

**Request**

```
GET /v1/face/group
```

**Response**
```
200 ok

{
    "groups": [
        <id>,
        ...
    ]
}
```

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|



## 1.4. 添加人脸
> 新增人脸 /face/group/add

**Request**

```
POST /v1/face/group/<id>/add
Content-Type: application/json

{
	"data": [
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"name": <name>,
				"pts": [[1213,400],[205,400],[205,535],[1213,535]]
			}
		},
		{
			"uri": "http://xx.com/xxx",
			"attribute": {
				"name": <name>
			}
		}
	]
}
```

Response

```
200 ok

Content-Type: application/json

{
	"faces": [
		<face_id>,
		...
	]
}

```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|
|uri|string|图片资源地址，其中包含且只包含一张人脸|
|name|string|人物姓名|
|pts|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|

## 1.5. 删除人脸
> 删除人脸特征 /face/group/delete

**Request**

```
POST /v1/face/group/<id>/delete
Content-Type: application/json

{
	"faces": [
		<face_id>,
        ...
	]
}
```

Response

```
200 ok

```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸库的唯一标识|
|face_id|string|人脸唯一标识|

**注意**
删除多个人脸时，部分成功也返回200OK

## 1.6. 人脸搜索
> 人脸库搜索  /face/group/search

Request

**json模式**
```
POST /v1/face/group/<group_ids>/search
Content-Type: application/json

{
	"data": {
		"uri": "http://xx.com/xxx",
	},
	"params": {
		"search_limit": 5,
		"face_limit": 5
	}
}
```

**octet-stream模式**
```
POST /v1/face/group/<group_ids>/search
Content-Type: application/octet-stream

<Body>

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
				"boundingBox":{
					"pts": [[1213,400],[205,400],[205,535],[1213,535]],
					"score":0.998
				},
				"groups": [
					{
						"values": [
							{
								"name": "xx",
								"id": "xxxx",
								"score":0.9998
							},
							...
						],
						"id": <group_id>,
						"type": 0
					}
				]
			}
		]
	}
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|group_ids|string|N个人脸库，逗号为分隔符|
|uri|string|图片资源地址|
|face_limit|int|最大单张图片检测人脸数目，默认face_limit=5|
|search_limit|int|最大相似人脸数，TOP N的搜索结果，默认search_limit=1|
|groups|string数组|期望搜索的人脸库id|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|boundingBox|map|人脸边框信息|
|boundingBox.pts|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|groups|object数组|每个人脸库的搜索结果|
|groups.id|string|人脸库id|
|groups.values.id|string|检索得到的人物ID|
|groups.value.name|string|检索得到的人物姓名|
|groups.value.score|float|0~1,检索结果的可信度，1为确定，人脸以可信度的降序排列|
|groups.type|int|搜索库类别，type=0表示默认库类型|

# 2. 人脸聚类排重API
> 目前采用1：N搜索模式

## 2.1. 创建人脸聚类库
> 新建人脸聚类库 /face/cluster/new

**Request**

```
 POST /v1/face/cluster/<id>/new
 Content-Type: application/json
 {
     "size": 100000,
     "precision": 4,
     "dimension": 512,
     "timeout": 600
 }
```

**Respose**

```
200 ok
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸聚类库的唯一标识|
|size|int|人脸库聚类预期特征数目，必填|
|precision|int|聚类特征精度，选填，默认precision=4，即float32|
|dimension|int|聚类特征维度，选填，默认dimension=512|
|timeout|int|聚类库内特征生存期，超时自动删除，选填，单位为秒，默认为0，表示永久存在|

## 2.2. 删除人脸聚类库
> 删除人脸聚类库 /face/cluster/remove

**Request**

```
 POST /v1/face/cluster/<id>/remove

```

**Respose**

```
200 ok
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|人脸聚类库的唯一标识|

## 2.3. 人脸聚类搜索
> 人脸聚类搜索请求  /face/cluster/search

Request

**json模式**
```
POST /v1/face/cluster/<group_ids>/search
Content-Type: application/json

{
	"data": {
		"uri": "http://xx.com/xxx",
	},
	"params": {
		"search_limit": 5,
		"face_limit": 5
	}
}
```

**octet-stream模式**
```
POST /v1/face/group/<group_ids>/search
Content-Type: application/octet-stream

<Body>

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
				"boundingBox":{
					"pts": [[1213,400],[205,400],[205,535],[1213,535]],
					"score":0.998
				},
				"groups": [
					{
						"values": [
							{
								"name": "xx",
								"id": "xxxx",
								"score":0.9998
							},
							...
						],
						"id": <group_id>，
						"type": 1
					}
				]
			}
		]
	}
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|group_ids|string|1+N个库ID，其中起一个为聚类库，后续为人脸搜索库，逗号为分隔符，N>=0|
|uri|string|图片资源地址|
|face_limit|int|最大单张图片检测人脸数目，默认face_limit=5|
|search_limit|int|最大相似人脸数，TOP N的搜索结果，默认search_limit=1|
|groups|string数组|期望搜索的人脸库id|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|boundingBox|map|人脸边框信息|
|boundingBox.pts|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|groups|object数组|每个人脸库的搜索结果|
|groups.id|string|库id|
|groups.values.name|string|检索得到的人物姓名，聚类人脸姓名为空|
|groups.values.id|string|检索得到的人物ID,首次发现的聚类人脸会自动分配ID|
|groups.values.score|float|0~1,检索结果的可信度，1为确定，人脸以可信度的降序排列|
|groups.type|int|搜索库类别，type=1表示类型为聚类库|

**注意**
* 聚类搜索接口，将搜索和聚类进行了结合，其逻辑为：先按N个人脸库进行搜索，若任何一个库命中，则不聚类，直接返回结果；否则返回聚类结果
* 属于同一个人脸的聚类结果groups.values.id相同
* 聚类库不会自动销户，需要手动创建和删除

# 3. 视频流处理服务API

## 3.1. 启动视频流处理服务
> 启动视频流处理服务 /camera/<id>/start

**Request**

```
 POST /v1/camera/<id>/start 
 Content-Type: application/json
 {
    "upstream_address": "rtsp://xxxxxxx",
    "groups": [
        "group1",
        "group2",
    ],
    "check_interval": 10,
    "enable_cluster_search": true,
    "cluster_feature_timeout": 0,
    "force":  true
 }
```

**Respose**

```
200 ok
{
    "downstream_address": "xxxxxxxxxxxxxxx",
    "cluster_id": "xxxxxxx"
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|摄像头的唯一标识, 必须为3至32位长度的字母与数字的组合|
|upstream_address|string|视频流的上行地址|
|groups|[]string|检索人脸库的group_id|
|check_interval|int [default:10]|抽帧的帧间隔,即每N帧抽一帧|
|enable_cluster_search|bool [default:false]|是否启动人脸聚类索引|
|cluster_feature_timeout|int|聚类库内特征生存期，超时自动删除，选填，单位为秒，默认为0，表示永久存在, (仅当启动人脸聚类索引时有效)|
|force|bool [default:false]|是否强制启动, 若强制, 则关闭之前的服务重新启动服务, 其余参数不用考虑之前的状态|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|downstream_address|string|视频流的下行地址|
|cluster_id|string [optional]|人脸聚类库id, 当且仅当启动聚类时才会返回此id|

## 3.2. 停止视频流处理服务
> 停止视频流处理服务 /camera/<id>/stop

**Request**

```
 POST /v1/camera/<id>/stop

```

**Respose**

```
200 ok
{
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|id|string|摄像头的唯一标识|
