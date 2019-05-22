<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [API定义](#api%E5%AE%9A%E4%B9%89)
  - [基本规格](#%E5%9F%BA%E6%9C%AC%E8%A7%84%E6%A0%BC)
  - [Service API](#service-api)
  - [API Specifications](#api-specifications)
    - [bjrun/politician/add](#bjrunpoliticianadd)
    - [bjrun/politician/list](#bjrunpoliticianlist)
    - [bjrun/politician/images](#bjrunpoliticianimages)
    - [bjrun/politician/del](#bjrunpoliticiandel)
    - [bjrun/politician/search](#bjrunpoliticiansearch)
    - [bjrun/image/add](#bjrunimageadd)
    - [bjrun/image/list/<label>](#bjrunimagelistlabel)
    - [bjrun/image/labels](#bjrunimagelabels)
    - [bjrun/image/del](#bjrunimagedel)
    - [bjrun/image/search](#bjrunimagesearch)
    - [bjrun/terror](#bjrunterror)
    - [pulp](#pulp)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# API定义

## 基本规格

HOST: argus.atlab.ai </br>
图片格式支持: `JPG`、`PNG`、`BMP`</br>
资源表示方式（URI）。通过统一方式定位、获取资源（图片、二进制数据等)

* HTTP， 网络资源，形如：`http://host/path`、`https://host/path`
* Data，Data URI Scheme形态的二进制文件，形如：`data:application/octet-stream;base64,xxx`。ps: 当前只支持前缀为`data:application/octet-stream;base64,`的数据

## Service API
| PATH | Note | Input | Response Type |
| :--- | :--- | :--- | :---: |
| [`/v1/bjrun/politician/add`](#bjrun/politician/add)|添加政治人物|object|json|
| [`/v1/bjrun/politician/list`](#bjrun/politician/list)|政治人物列表|GET|json|
| [`/v1/bjrun/politician/images`](#bjrun/politician/images)|政治人物图片列表|Plitician Name|json|
| [`/v1/bjrun/politician/del`](#bjrun/politician/del)|删除政治人物|object|json|
| [`/v1/bjrun/politician/search`](#bjrun/politician/search)|政治人物检索|object|json|
| [`/v1/bjrun/image/add`](#bjrun/image/add)|增加图片信息|image URI array|json|
| [`/v1/bjrun/image/list/<label>`](#bjrun/image/list/<label>)|查询图片列表|GET|json|
| [`/v1/bjrun/image/labels`](#bjrun/image/labels)|查询图片列表|GET|json|
| [`/v1/bjrun/image/del`](#bjrun/image/del)|删除图片信息|image URI array|json|
| [`/v1/bjrun/image/search`](#bjrun/image/search)|图片检索|image URI array|json|
| [`/v1/bjrun/terror`](#bjrun/terror)|暴恐识别|image URI|json|
| [`/v1/pulp`](#pulp)|融合剑皇|image URI|json|


## API Specifications


### bjrun/politician/add

> 输入领导人姓名和图片地址列表，返回API调用状态和添加失败的图片

Request

```
POST /v1/bjrun/politician/add  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
		  "attribute":{
		       "name":"Roby"
		  }
	   },
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
		   "attribute":{
		       "name":"Alice"
		  }
	   },
	   ...
	]
}
```

Response

```
200 ok

{
	"code": 0,
	"message": "success"
	"result":{
	   "failed":["http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",...]
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|attribute.name|string|政治人物姓名|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确, |
|message|string|结果描述信息|
|result.failed|list|添加失败的图片列表,失败原因包括下载失败，无人脸或多个人脸等|

### bjrun/politician/list

> 返回API调用状态和政治人物列表

Request

```
GET /v1/bjrun/politician/list  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>
```

Response

```
200 ok

{
	"code": 0,
	"message": "success"
	"result":{
	   "politicians":["Alice","Alizee","Jack.Ma","习近平"...]
	}
}
```

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确, |
|message|string|结果描述信息|
|result.politicians|list|政治人物姓名列表|

### bjrun/politician/images

> 输入政治人物人姓名，返回对应人物所有图片URL列表

Request

```
POST /v1/bjrun/politician/images  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"params": {
	   "name":"Alice"
	}
}
```

Response

```
200 ok

{
	"code": 0,
	"message": "success"
	"result":{
	   "images":[url1,url2...],
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|params.name|string|政治人物姓名|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确, |
|message|string|结果描述信息|
|result.images|list|图片列表|

### bjrun/politician/del

> 输入政治人物姓名和图片列表，返回API调用状态

Request

```
POST /v1/bjrun/politician/del  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
		  "attribute":{
               "name":"Roby"
          }
	   },
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
		  "attribute":{
               "name":"Roby"
          }
	   }
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
|uri|string|图片资源地址,若data中不含uri,则删除该领导所有图片|
|name|string|政治人物姓名|


### bjrun/politician/search

> 政治人物搜索，对输入图片识别检索是否存在政治人物

Request

```
POST bjrun/politician/search Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://xx.com/xxx"
	},
	"params":{
		"filters":["Roby","王岐山"]
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
|data.filters|list|人物过滤列表|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|review|boolean|True或False,图片是否需要人工review, 只要有一个value.review为True,则此字段为True|
|boundingBox|map|人脸边框信息|
|boundingBox.pst|list[4]|人脸边框在图片中的位置[左上，右上，右下，左下]|
|boundingBox.score|float|人脸位置检测准确度|
|value.name|string|检索得到的政治人物姓名,value.score < 0.525时未找到相似人物,没有这个字段|
|value.review|boolean|True或False,当前人脸识别结果是否需要人工review|
|value.score|float|0~1,检索结果的可信度, 0.4 <= value.score <=0.6 时 value.review 为True|
|sample|object|该政治人物的示例图片信息，value.score < 0.525时未找到相似人物, 没有这个字段|
|sample.url|string|该政治人物的示例图片|
|sample.pts|list[4]|人脸在示例图片中的边框|


### bjrun/image/add

> 图片地址列表，返回API调用状态码

Request

```
POST /v1/bjrun/image/add  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
		  "attribute":{
		    "label":"label1"
		  }
	   },
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
		  "attribute":{
		    "label":"label2"
		  }
	   }
	]
}
```

Response

```
200 ok

{
	"code": 0,
	"message": "success"
	"result":{
	   "failed":["http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",...]
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|data.uri|string|图片资源地址|
|data.label|string|图片类别|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确, |
|message|string|结果描述信息|
|result.failed|list|添加失败的图片列表|

### bjrun/image/list/<label>

> 返回某个label下面的所有图片

Request

```
Get /v1/bjrun/image/list/<label>  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 ok

{
	"code": 0,
	"message": "success"
	"result":{
	   "images":["http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",...]	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|label|string|图片类别|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确, |
|message|string|结果描述信息|
|result.images|list|图片列表|


### bjrun/image/labels

> 返回所有的label(图片类别)

Request

```
Get /v1/bjrun/image/labels  Http/1.1
Authorization: Qiniu <AccessKey>:<Sign>

```

Response

```
200 ok

{
	"code": 0,
	"message": "success"
	"result":{
	   "labels":["label1",“label2”，...]
    }
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|page|int|数据检索起始页,从0开始|
|size|int|数据检索页面大小,小于2000|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确, |
|message|string|结果描述信息|
|result.labels|list|图片类别列表|


### bjrun/image/del

> 输入图片地址列表，返回API调用状态码

Request

```
POST /v1/bjrun/image/del Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
		  "attribute":{
               "label":"Alice"
          }
	   },
	   {
		  "uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg",
		  "attribute":{
               "label":"Alice"
          }
	   }
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
|uri|string|图片资源地址,若uri为空则删除当前label下的所有图片信息|
|params.label|string|所属类别|

### bjrun/image/search

> bjrun特定图片检索

Request

```
POST /v1/bjrun/image/search  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://xx.com/xxx"
	},
	"params": {
		"limit": 1,
		"filters":["landscape","ocean"]
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
			"url": "xx",
			"label":"label1",
			"score":0.9998
		},
		...
	]
}
```
请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|params.filters|list|图片label过滤列表|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|url|string|检索得到的示例图片地址|
|score|float|0~1,检索结果的可信度，1为确定|


### bjrun/terror

> 用检测暴恐识别和分类暴恐识别方法做融合暴恐识别<br>
> 每次输入一张图片，返回其内容是否含暴恐信息

Request

```
POST /v1/bjrun/terror  Http/1.1
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
		"class": "guns",
		"review":false
	}
}
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址, Http资源或Data URI scheme 二进制数据|

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示正确|
|message|string|结果描述信息|
|result.label|int|标签{0:正常，1:暴恐}|
|result.class|string|精确的暴恐类别信息|
|result.score|float|该图像被识别为某个分类的概率值，概率越高、机器越肯定|
|result.review|bool|是否需要人工review|


### pulp

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