<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [基本参数](#%E5%9F%BA%E6%9C%AC%E5%8F%82%E6%95%B0)
  - [输入图片格式](#%E8%BE%93%E5%85%A5%E5%9B%BE%E7%89%87%E6%A0%BC%E5%BC%8F)
- [API](#api)
  - [detect](#detect)
  - [face/cluster](#facecluster)
  - [face/detect](#facedetect)
  - [face/search/politician](#facesearchpolitician)
  - [search/politician](#searchpolitician)
  - [face/sim](#facesim)
  - [pulp](#pulp)
  - [pulp/recognition](#pulprecognition)
  - [scene](#scene)
  - [terror](#terror)
  - [terror/complex](#terrorcomplex)
  - [image/label](#imagelabel)
  - [image/censor](#imagecensor)
  - [ocr/text](#ocrtext)
  - [ocr/scene](#ocrscene)
  - [ocr/idcard](#ocridcard)
  - [ocr/vat](#ocrvat)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 基本参数

## 输入图片格式
支持`JPG`、`PNG`、`BMP`

# API

| PATH | Note | Input | Response Type |
| :--- | :--- | :--- | :---: |
| [`/v1/detect`](#detect) | 通用物体检测 | image URI | Json |
| [`/v1/face/cluster`](#face/cluster)|人脸聚类|image URI Array|json|
| [`/v1/face/detect`](#face/detect) | 人脸位置、年龄和性别检测 | image URI | Json |
| [`/v1/face/search/politician`](#face/search/politician) | 政治人物搜索 | image URI | json |
| [`/v1/search/politician`](#search/politician) | 政治人物搜索,可加limit参数 | image URI | json |
| [`/v1/face/sim`](#face/sim)|人脸相似性检测|image URI Array|json|
| [`/v1/pulp`](#pulp)|融合剑皇|image URI|json|
| [`/v1/pulp/recognition`](#pulp/recognition)|融合剑皇v0版|image URI Array|json|
| [`/v1/scene`](#scene) | 通用场景识别 | image URI | Json |
| [`/v1/terror`](#terror)|暴恐识别|image URI|json|
| [`/v1/image/label`](#image/label)|图片打标|image URI|json|
| [`/v1/image/censor`](#image/censor)|图片审核|image URI|json|
| [`/v1/ocr/text`](#ocr/text)|OCR文本识别|image URI|json|
| [`/v1/ocr/scene`](#ocr/scene)|通用OCR文本识别|image URI|json|
| [`/v1/ocr/idcard`](#ocr/idcard)|身份证识别|image URI|json|
| [`/v1/ocr/vat`](#ocr/vat)|增值税发票识别|image URI|json|


## detect

> 通用物体检测，简单转发eval的请求

Request

```
POST /v1/detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://o8smkso2w.bkt.clouddn.com/dog.jpg"
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|


Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"detections": [
			{
				"class": "dog",
				"index": 58,
				"pts": [[138,200],[305,200],[305,535],[138,535]],
				"score": 0.9842000007629395
			},
			...
		]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|index|uint|物体分类序号|
|class|string|物体分类名称|
|score|float|物体检测的准确度，取值范围0~1，1为准确度最高|
|pts|四点坐标值|[左上，右上，右下，左下]四点坐标框定的物体|


## face/cluster

> 检测人脸并聚类<br>
> 每次输入一组图片，返回人脸框和所属类别

Request

```
POST /v1/face/cluster  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
		{
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
		},
		{
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
		}
	]
}
```

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"cluster": [
			[//每张图片中所有人脸在一个list中
				{
					"boundingBox":{
						"pts": [[138,200],[305,200],[305,535],[138,535]],
						"score":0.9998
					},
					"group":{
						"id": 1,
						"center_dist": 0.0
					}
				}
			],
			[
				{
					"boundingBox":{
						"pts": [[1213,400],[205,400],[205,535],[1213,535]],
						"score":0.9998
					},
					"group":{
						"id": 1,
						"center_dist": 0.156
					}
				}
			] 
		]
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|boundingBox|map|人脸边框信息|
|boundingBox.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|boundingBox.score|float|人脸位置检测准确度|
|group|map|人脸聚类信息|
|group.id|int|人脸所属类别号,-1表示独自是一个组|
|group.center_dist|float|人脸到其聚类中心的距离|


## face/detect

> 检测人脸所在位置，对应人脸的年龄以及性别<br>
> 每次输入一张图片，返回所有检测到的脸的位置、年龄和性别

Request

```
POST /v1/face/detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
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
				"bounding_box":{
					"pts": [[138,200],[305,200],[305,535],[138,535]],
					"score":0.9998
				},
				"age":{
					"value": 26.87,
					"score":0.9876
				},
				"gender":{
					"value":"Female",
					"score":0.9657
				}
			}
		]
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|bounding_box|map|人脸边框信息|
|bounding_box.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|bounding_box.score|float|人脸位置检测准确度|
|age|map|年龄检测信息1-100|
|age.value|float|人脸年龄|
|age.score|float|人脸年龄检测准确度|
|gender|map|性别检测|
|gender.value|string|预测性别,Male 或Female|
|gender.score|float|人脸性别检测准确度|


## face/search/politician

> 政治人物搜索，对输入图片识别检索是否存在政治人物

Request

```
POST /v1/face/search/politician Http/1.1
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
		"review": True,
		"detections": [
			{
				"boundingBox":{
					"pts": [[1213,400],[205,400],[205,535],[1213,535]],
					"score":0.998
				},
				"value": {
					"name": "xx",
					"group": "Inferior Artist",
					"score":0.567,
					"review": True
				},
				"sample": {
					"url": "",
					"pts": [[1213,400],[205,400],[205,535],[1213,535]]
				}
			},
			{
				"boundingBox":{
					"pts": [[1109,500],[205,500],[205,535],[1109,535]],
					"score":0.98
				},
				"value": {
					"score":0.987,
					"review": False
				}
			}
		]
	}
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|review|boolean|True或False,图片是否需要人工review, 只要有一个value.review为True,则此字段为True|
|boundingBox|map|人脸边框信息|
|boundingBox.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|boundingBox.score|float|人脸位置检测准确度|
|value.name|string|检索得到的政治人物姓名,value.score < 0.4时未找到相似人物,没有这个字段|
|value.group|string|人物分组信息，总共有7个组{'Domestic politician','Foreign politician','Sacked Officer(Government)','Sacked Officer (Enterprise)',
'Anti-China Molecule','Terrorist','Inferior Artist'}|
|value.review|boolean|True或False,当前人脸识别结果是否需要人工review|
|value.score|float|0~1,检索结果的可信度, 0.35 <= value.score <=0.45 时 value.review 为True|
|sample|object|该政治人物的示例图片信息，value.score < 0.4时未找到相似人物, 没有这个字段|
|sample.url|string|该政治人物的示例图片|
|sample.pts|list[4]|人脸在示例图片中的边框|

## search/politician

> 政治人物搜索，对输入图片识别检索是否存在政治人物

Request

```
POST /v1/search/politician Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://xx.com/xxx"
	},
	"params": {
		"limit": 2
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
		"review": True,
		"detections": [
			{
				"boundingBox":{
					"pts": [[1213,400],[205,400],[205,535],[1213,535]],
					"score":0.998
				},
				"politician": [
					{
						"name": "习近平",
						"group": "Sacked Officer(Government)",
						"score": 0.84919655,
						"review": false,
						"sample": {
							"url":"http://p4wgdf8lr.bkt.clouddn.com/sample.jpg",
							"pts": [[215,74],[326,74],[326,209],[215,209]]
						}
					},
					{
						"name": "习近平",
						"score": 0.84149814,
						"review": false,
						"sample": {
							"url": "http://p4wgdf8lr.bkt.clouddn.com/8.jpg",
							"pts": [[249,176],[377,176],[377,335],[249,335]]
						}
					}]
			},
			{
				"boundingBox":{
					"pts": [[1109,500],[205,500],[205,535],[1109,535]],
					"score":0.98
				},
				"politician": [
				{
					"name": "张德江",
					"group": "Sacked Officer(Government)",
					"score": 0.9149814,
					"review": false,
					"sample": {
						"url": "http://p4wgdf8lr.bkt.clouddn.com/s8.jpg",
						"pts": [[249,176],[377,176],[377,335],[249,335]]
					}
				}]
			}
		]
	}
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|limit|int||

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|review|boolean|True或False,图片是否需要人工review, 只要有一个value.review为True,此字段为True|
|boundingBox|map|人脸边框信息|
|boundingBox.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|boundingBox.score|float|人脸位置检测准确度|
|politician|list|检索到的人物列表||
|politician.name|string|检索得到的政治人物姓名,value.score < 0.66时未找到相似人物,没这个字段|
|politician.group|string|人物分组信息，总共有7个组{'Domestic politician','Foreign politician','Sacked Officer(Government)','Sacked Officer (Enterprise)',
'Anti-China Molecule','Terrorist','Inferior Artist'}|
|politician.review|boolean|True或False,当前人脸识别结果是否需要人工review|
|politician.score|float|0~1,检索结果的可信度|
|politician.sample|object|该政治人物的示例图片信息|
|politician.sample.url|string|该政治人物的示例图片|
|politician.sample.pts|list[4]|人脸在示例图片中的边框|

## face/sim

> 人脸相似性检测<br>
> 若一张图片中有多个脸则选择最大的脸

Request

```
POST /v1/face/sim  Http/1.1
Content-Type:application/json
Authorization:Qiniu <AccessKey>:<Sign>

{
	"data": [{
		"uri": "http://image2.jpeg"
	},{
		"uri": "http://image1.jpeg
	}]
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|data|list|两个图片资源地址|


Response

```
200 ok

{
	"code": 0,
	"message": "success",
	"result": {
		"faces":[{
				"score": 0.987,
				"pts": [[225,195], [351,195], [351,389], [225,389]]
			},
			{
				"score": 0.997,
				"pts": [[225,195], [351,195], [351,389], [225,389]]
			}], 
		"similarity": 0.87,
		"same": 0  
	}	
}	
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|faces|list|两张图片中选择出来进行对比的脸|
|score|float|人脸识别的准确度，取值范围0~1，1为准确度最高|
|pts|list|人脸在图片上的坐标|
|similarity|float|人脸相似度，取值范围0~1，1为准确度最高|
|same|bool|是否为同一个人|


## pulp

> 用第三方服务和AtLab的剑皇服务做融合剑皇<br>
> 每次输入一张图片，返回其内容是否含色情信息

Request

```
POST /v1/pulp  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
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
		"label":1,
		"score":0.987,
		"review":false
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result.label|int|标签{0:色情，1:性感，2:正常}|
|result.score|float|色情识别准确度|
|result.review|bool|是否需要人工review|


## pulp/recognition

> 完全实现ataraxia v0 版线上剑皇接口，以无缝迁移老用户到ava 

Request 

```
POST /v1/pulp/recognition
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	image:[
		"url1", 
		"url2"
		...
		]	
}

```

Response

```
200 ok

{  
	"code": 0,
	"message": "success",	
	"nonce": “324443444”,	
	"timestamp": 149606905955，
	"pulp"：{
		“reviewCount”: 1, 
		"statistic": [1, 1, 0],
		"fileList": [
			{
				"result":{
					"rate": 0.987, 
					"label": 0,  
					"name":  "url1",  
					"review": False
				}
			},
			{
				"result":{
					"rate": 0.708, 
					"label": 1,  
					"name":  "url2",  
					"review": True
				}
			}
		]
	}
}

```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|image|list|图片资源地址列表|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|nonce|string|Int32 类型的随机整数字符串|
|timestamp|uint64|linux时间戳|
|pulp.reviewCount|int|需要人工review的图片数目|
|pulp.statistic|list[3]|{pulp,sexy,norm}各类型图片数目统计|
|pulp.fileList|list|结果列表|
|pulp.fileList.result.rate|float|0～1 结果准确度|
|pulp.fileList.result.label|int|{-1,0,1,2}代表图片所属类别{error,pulp,sexy,norm},-1:检测出错|
|pulp.fileList.result.name|string|图片url|
|pulp.fileList.result.review|boolean|True:需要人工review,False:不需要人工review|


## scene

> 通用场景识别

Request

```
POST /v1/scene  Http/1.1
Content-Type:application/json
Authorization:Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg"
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|


Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"confidences": [
			{
				"class": "/v/valley",
				"index": 345,
				"label": ["outdoor","landscape"],
				"score": 0.3064107298851013
			}
		]		
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|index|uint|场景识别分类序号|
|class|string|场景识别分类名称|
|lable|[string]|场景识别分类标签组|
|score|float|场景识别的准确度，取值范围0~1，1为准确度最高|


## terror

> 用检测暴恐识别和分类暴恐识别方法做融合暴恐识别<br>
> 每次输入一张图片，返回其内容是否含暴恐信息

Request

```
POST /v1/terror  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
	},
	"params": {
		"detail": true
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
		"label":1,
		"score":0.987,
		"review":false
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|params.detail|bool|是否显示详细信息；可选参数|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result.label|int|标签{0:正常，1:暴恐}|
|result.class|string|标签（指定`detail=true`的情况下返回）|
|result.score|float|暴恐识别准确度|
|result.review|bool|是否需要人工review|

## terror/complex

> 融合暴恐识别和暴恐识别结果进行暴恐鉴别<br>
> 每次输入一张图片，返回其内容是否含暴恐信息，同时含有暴恐分类和暴恐检测结果

Request

```
POST /v1/terror/complex  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
	},
	"params": {
		"detail": true
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
		"label":1,
		"score":0.987,
		"review":false
		"classes": [
			{
				"class":"bomb_fire",
				"score": 0.97
			},
			{
				"class": "guns",
				"score": 0.95
			}
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|params.detail|bool|是否显示详细信息；可选参数|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result.label|int|标签{0:正常，1:暴恐}|
|result.classes.class|string|标签类别（指定`detail=true`的情况下返回）|
|result.classes.score|float32|标签类别准确度（指定`detail=true`的情况下返回）|
|result.score|float|暴恐识别准确度，取所有标签中最高准确度|
|result.review|bool|是否需要人工review|

## image/label

> 365类场景分类、545类物体检测和1000类目标识别融合而成的图片打标<br>
> 每次输入一张图片，尽可能多而准确的返回图片中的主要场景、物体、动物等内容

Request

```
POST /v1/image/label  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
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
		"confidences": [
			{
				"class": "/s/shower",
				"score": 0.8726697,
				"label_cn": "xx"
			},
			{
				"class": "Woman",
				"score": 0.6530496,
				"label_cn": "xx"
			},
			...
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|confidences|list|图片打标信息列表|
|class|string|识别到的物体或场景名称|
|score|float|所识别内容的准确度|
|label_cn|string|中文标签|


## image/censor

> 图片审核<br>

Request

```
POST /v1/image/censor Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
	},
	"params": {
		"type": [
			"pulp",
			"terror",
			"politician",
			"terror-complex"
		],
		"detail": true
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
		"label": 1,
		"score": 0.888,
		"details": [
			{
				"type": "pulp",
				"label": 1,
				"score": 0.8726697,
				"review": false
			},
			{
				"type": "terror",
				"label": 1,
				"class": <class>,
				"score": 0.6530496,
				"review": false
			},
			{
				"type": "politician",
				"label": 1,
				"score": 0.77954,
				"review": True,
				"more": [
					{
						"boundingBox":{
							"pts": [[1213,400],[205,400],[205,535],[1213,535]],
							"score":0.998
						},
						"value": {
							"name": "xx",
							"score":0.567,
							"review": True
						},
						"sample": {
							"url": "",
							"pts": [[1213,400],[205,400],[205,535],[1213,535]]
						}
					},
					{
						"boundingBox":{
							"pts": [[1109,500],[205,500],[205,535],[1109,535]],
							"score":0.98
						},
						"value": {
							"score":0.987,
							"review": False
						}
					}
				]
			},
			{
				"type": "terror-complex",
				"label": 1,
				"classes": <classes>,
				"score": 0.6530496,
				"review": false
			},
		]
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|params.type|string|选择的审核类型，可选项："pulp"/"terror"/"politician"/"terror-complex"；可选参数，不填表示全部执行|
|params.detail|bool|是否显示详细信息；可选参数|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result.label|int|是否违规，0：不违规；1：违规|
|result.score|float|是否违规置信度|
|result.detail.type|string|审核类型|
|result.detail.label|int|审核结果类别，具体看各类型|
|result.detail.class|string|详细类别，具体看各类型|
|result.detail.classes|[]string|详细类别列表，具体看各类型|
|result.detail.score|float|审核结果置信度|



## ocr/text

> OCR文本识别，主要针对微博、微信等长文本图片的识别

Request

```
POST /v1/ocr/text  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
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
		"type": "wechat",
		"bboxes": [
			[[140,1125],[596,1161],[140,1046],[331,1082]],
			[[140,1005],[563,1041],[141,1167],[238,1200]],
			[[140,924],[594,962],[141,237],[605,273]],
			...,
			[[119,182],[194,210],[119,502],[194,531]]
		],
		"texts": [
			'防治疗中有种非医果药物做贝',
			'抗。2万多一计副作册小发',
			'就开口了，跟勿说了你前天的瘤',
			...,
			'手术了，化疗吧。'
		]
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|type|string|文本类别，{"wechat","blog","other-text","normal"}，分别表示微信、微博、其他文本、非文本|
|bboxes|list[4,2]<int>|图片中所有的文本框位置，为顺时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|texts|list<string>|对应四边形框中的文字|



## ocr/scene

> 通用OCR文本识别

Request

```
POST /v1/ocr/scene  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
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
		"bboxes": [
			[140,1125,596,1161,140,1046,331,1082],
			[140,1005,563,1041,141,1167,238,1200],
			[140,924,594,962,141,237,605,273],
			...,
			[119,182,194,210,119,502,194,531]
		],
		"text": [
			'7月部天我么',
			'大学有这的艺术院系向其',
			'这世界总有那么一生',
			'资高家小蓝单车网来信',
			'一要际“我',
			'下见考创',
			'上大士联样呢',
			'想双动，杨又意'
		]
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|bboxes|[[四点坐标]]|图片中所有的文本框位置，为顺时针方向旋转的任意四边形[x1,y1,x2,y2,x3,y3,x4,y4]|
|text|string|对应四边形框中的文字|


## ocr/idcard

> 身份证识别（全字段正反面通用）

Request

```
POST /v1/ocr/idcard  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://p9zv90cqq.bkt.clouddn.com/001.jpg"
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
		"uri": "data:application/octet-stream;base64, ...",
		"bboxes": [
			[[134,227],[419,227],[419,262],[134,262]],
			...
			[[115,50],[115,100],[232,100],[232,50]]
		],
		"type": 0,
		"res": {
			"住址": "河南省项城市芙蓉巷东四胡同2号",
			"公民身份号码": "412702199705127504",
			"出生": "1997年5月12日",
			"姓名": "张杰",
			"性别": "女",
			"民族": "汉"
		}
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|返回码： 0表示成功，非0表示出错|
|message|string|结果描述信息|
|uri|string|截取原图中身份证区域后的图片 base64 编码|
|bboxes|list[4,2]<float>|返回的图片中所有的文本框位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|type|int|身份证正反面信息，0:正面，1:背面|
|res|map[string]string|识别后信息结构化的结果|
|res[姓名]|string|姓名|
|res[性别]|string|性别|
|res[民族]|string|民族|
|res[出生]|string|出生|
|res[住址]|string|住址|
|res[公民身份号码]|string|身份证号|
|res[有效期限]|string|有效期限|
|res[签发机关]|string|签发机关|


## ocr/vat

> 增值税发票识别（高研院版，全字段正反面通用）

Request

```
POST /v1/ocr/sari/vat  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://pbqb5ctvq.bkt.clouddn.com/YBZZS_01488003.jpg"
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
		"bboxes": [
			[[134,227],[419,227],[419,262],[134,262]],
			...
			[[115,50],[115,100],[232,100],[232,50]]
		],
		"res": {
			_BeiZhu: '',
			_DaLeiMingCheng: '',
			_DaiKaiBiaoShi: '',
			_DaiKaiJiGuanDiZhiJiDianHua: '',
			_DaiKaiJiGuanGaiZhang: '',
			_DaiKaiJiGuanHaoMa: '',
			_DaiKaiJiGuanMingCheng: '',
			_DanZhengMingCheng: '四川增值税专用发票',
			_FaPiaoDaiMa_DaYin: '51001xxxx',
			_FaPiaoDaiMa_YinShua: '51001xxxx0',
			_FaPiaoHaoMa_DaYin: '01488xxx',
			_FaPiaoHaoMa_YinShua: '0148xxx3',
			_FaPiaoJianZhiZhang: '',
			_FaPiaoLianCi: '抵扣联',
			_FaPiaoYinZhiPiHanJiYinZhiGongSi: '',
			_FuHeRen: '',
			_GaiZhangDanWeiMingCheng: '',
			_GaiZhangDanWeiShuiHao: '',
			_GouMaiFangDiZhiJiDianHua: 'xxx科创园区园艺东街x号xxxx-2536xx',
			_GouMaiFangKaiHuHangJiZhangHao: '中国银行股份有限公司xx支行 1239123xxxx',
			_GouMaiFangMingCheng: '四川xx有限公司',
			_GouMaiFangNaShuiShiBieHao: '510790567644042',
			_HeJiJinE_BuHanShui: '￥11230.77',
			_HeJiShuiE: '￥1909.2',
			_JiQiBianHao: '附',
			_JiaShuiHeJi_DaXie: 'ⓧ壹万叁仟壹佰肆拾圆整',
			_JiaShuiHeJi_XiaoXie: '￥13140.00',
			_JiaoYanMa: '',
			_KaiPiaoRen: '张xx',
			_KaiPiaoRiQi: '2015年12月15日',
			_MiMa: '47x<*51+5485+5634xxx135>85*/42xx3708x2**107*<65x90+09+5x25>54xx+*3151xxx381-1/4xx4-0xx74xx2*<9xx3',
			_ShouKuanRen: '',
			_WanShuiPingZhengHao: '',
			_XiaoLeiMingCheng: '一般增值税发票',
			_XiaoShouDanWeiGaiZhangLeiXing: '',
			_XiaoShouFangDiZhiJiDianHua: 'xx南路东段21世纪二期 0xxx-253xxxx',
			_XiaoShouFangKaiHuHangJiZhangHao: '工行绵阳临园支行23084124090248xxxxx',
			_XiaoShouFangMingCheng: '绵阳xxxx有限公司',
			_XiaoShouFangNaShuiRenShiBieHao: '51xxx8551xxxxx7',
			_XiaoShouMingXi: [Array]
		}
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|bboxes|list[4,2]<float32>|返回的图片中所有的文本框位置，为顺时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|res|string,list|识别后信息结构化的结果|

返回字段对照表：
|字段|说明|
|:---|:---|
|_BeiZhu|备注|
|_DaLeiMingCheng|大类名称|
|_DaiKaiBiaoShi|代开标识|
|_DaiKaiJiGuanDiZhiJiDianHua|代开机关地址及电话|
|_DaiKaiJiGuanGaiZhang|代开机关盖章|
|_DaiKaiJiGuanHaoMa|代开机关号码|
|_DaiKaiJiGuanMingCheng|代开机关名称|
|_DanZhengMingCheng|单证名称|
|_FaPiaoDaiMa_DaYin|发票代码 打印|
|_FaPiaoDaiMa_YinShua|发票代码 印刷|
|_FaPiaoHaoMa_DaYin|发票号码 打印|
|_FaPiaoHaoMa_YinShua|发票号码 印刷|
|_FaPiaoJianZhiZhang|发票监制章|
|_FaPiaoLianCi|发票联|
|_FaPiaoYinZhiPiHanJiYinZhiGongSi|发票印制批号及印制公司|
|_FuHeRen|复合人|
|_GaiZhangDanWeiMingCheng|盖章单位名称|
|_GaiZhangDanWeiShuiHao|盖章单位税号|
|_GouMaiFangDiZhiJiDianHua|购买方地址及电话|
|_GouMaiFangKaiHuHangJiZhangHao|购买方开户行及账号|
|_GouMaiFangMingCheng|购买方名称|
|_GouMaiFangNaShuiShiBieHao|购买方纳税识别号|
|_HeJiJinE_BuHanShui|合计金额 - 不含税|
|_HeJiShuiE|合计税额|
|_JiQiBianHao|机器编号|
|_JiaShuiHeJi_DaXie|加税合计 - 大写|
|_JiaShuiHeJi_XiaoXie|加税合计 - 小写|
|_JiaoYanMa|校验码|
|_KaiPiaoRen|开票人|
|_KaiPiaoRiQi|开票日期|
|_MiMa|密码|
|_ShouKuanRen|收款人|
|_WanShuiPingZhengHao|完税凭证号|
|_XiaoLeiMingCheng|小类名称|
|_XiaoShouDanWeiGaiZhangLeiXing|销售单位盖章类型|
|_XiaoShouFangDiZhiJiDianHua|销售方地址及电话|
|_XiaoShouFangKaiHuHangJiZhangHao|销售方开户行及账号|
|_XiaoShouFangMingCheng|销售方名称|
|_XiaoShouFangNaShuiRenShiBieHao|销售方纳税人识别号|
|_XiaoShouMingXi|销售明细|
