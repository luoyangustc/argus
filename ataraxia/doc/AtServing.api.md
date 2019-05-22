<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [基本参数](#%E5%9F%BA%E6%9C%AC%E5%8F%82%E6%95%B0)
  - [输入图片格式](#%E8%BE%93%E5%85%A5%E5%9B%BE%E7%89%87%E6%A0%BC%E5%BC%8F)
- [Cmd API](#cmd-api)
  - [pulp](#pulp)
  - [terror-classify](#terror-classify)
  - [terror-detect](#terror-detect)
  - [facex-age](#facex-age)
  - [facex-gender](#facex-gender)
  - [facex-detect](#facex-detect)
  - [detection](#detection)
  - [object-classify](#object-classify)
  - [scene](#scene)
  - [facex-feature](#facex-feature)
  - [segment](#segment)
  - [facex-cluster](#facex-cluster)
  - [politician](#politician)
  - [search-bjrun](#search-bjrun)
  - [ocr-idcard](#ocr-idcard)
  - [image-feature](#image-feature)
  - [video-feature](#video-feature)
  - [video-classify](#video-classify)
  - [ocr-classify](#ocr-classify)
  - [ocr-detect](#ocr-detect)
  - [ocr-recognize](#ocr-recognize)
  - [ocr-scene-detect](#ocr-scene-detect)
  - [ocr-scene-recog](#ocr-scene-recog)
  - [police-detect](#police-detect)
  - [ocr-sari-crann](#ocr-sari-crann)
  - [ocr-sari-id-pre](#ocr-sari-id-pre)
  - [ocr-sari-vat](#ocr-sari-vat)
  - [ocr-ctpn](#ocr-ctpn)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 基本参数

## 输入图片格式
支持`JPG`、`PNG`、`BMP`

# Cmd API

Request `POST /v1/eval/<cmd>[/<version>]`

| PATH | Note | Input | Response Type |
| :--- | :--- | :--- | :---: |
| [`/v1/eval/pulp`](#pulp) | 剑皇 | image URI | Json |
| [`/v1/eval/pulp-small`](#pulp-small) | 剑皇分类小模型 | image URI | Json |
| [`/v1/eval/pulp-detect`](#pulp-detect) | 剑皇检测 | image URI | Json |
| [`/v1/eval/terror-classify`](#terror-classify) | 暴恐检测 | image URI | Json |
| [`/v1/eval/terror-detect`](#terror-detect) | 暴恐检测 | image URI | Json |
| [`/v1/eval/facex-age`](#face-age) | 人脸年龄检测 | image URI, Rect Pts | Json |
| [`/v1/eval/facex-gender`](#face-gender) | 人脸性别检测 | image URI, Rect Pts | Json |
| [`/v1/eval/facex-detect`](#facex_detect) | 人脸检测 | image URI | Json |
| [`/v1/eval/facex-pose`](#facex_pose) | 人脸姿态 | image URI | Json |
| [`/v1/eval/detection`](#detection) | 物体检测 | image URI | Json |
| [`/v1/eval/object-classify`](#object-classify) | 物体识别 | image URI | Json |
| [`/v1/eval/scene`](#scene) | 场景识别 | image URI | Json |
| [`/v1/eval/facex-feature`](#facex-feature) | 人脸特征提取 | image URI, Rect Pts | Stream |
| [`/v1/eval/segment`](#segment) | 物体分割 | image URI | Json |
| [`/v1/eval/facex-cluster`](#facex-cluster) | 人脸聚类 | image URI, Rect Pts | Json |
| [`/v1/eval/politician`](#politician) | 政治人物检索 | image URI | Json |
| [`/v1/eval/search-bjrun`](#search-bjrun) | bjrun特定图片库检索 | image URI | Json |
| [`/v1/eval/ocr-idcard`](#ocr-idcard) | ocr身份证识别 | image URI | Json |
| [`/v1/eval/image-feature`](#image-feature) | 图片特征提取 | image URI | Json |
| [`/v1/eval/video-feature`](#video-feature) | 视频特征提取(cs,私有部署) | image URI | Json |
| [`/v1/eval/video-classify`](#video-classify) | 视频分类(cs,私有部署) |  image features | Json |
| [`/v1/eval/ocr-classify`](#ocr-classify) | ocr图片分类 |  image URI | Json |
| [`/v1/eval/ocr-detect`](#ocr-detect) | ocr图片文字检测 |  image URI | Json |
| [`/v1/eval/ocr-recognize`](#ocr-recognize) | ocr文字图片识别 |  image URI | Json |
| [`/v1/eval/ocr-scene-detect`](#ocr-scene-detect) | 通用ocr文字检测 |  image URI | Json |
| [`/v1/eval/ocr-scene-recog`](#ocr-scene-recog) | 通用ocr文字识别 |  image URI | Json |
| [`/v1/eval/ocr-sari-crann`](#ocr-sari-crann) | crann文字识别 |  image URI | Json |
| [`/v1/eval/ocr-sari-id-pre`](#ocr-sari-id-pre) | 身份证识别预处理 |  image URI | Json |
| [`/v1/eval/ocr-sari-vat`](#ocr-sari-vat) | 增值税发票文字检测及结构化后处理 |  image URI | Json |
| [`/v1/eval/ocr-ctpn`](#ocr-ctpn) | 长文本文字检测 |  image URI | Json |
| [`/v1/eval/ads-detection`](#ads-detection) | 商业广告检测 |  image URI | Json |
| [`/v1/eval/ads-recognition`](#ads-detection) | 商业广告检测 |  json  | Json |
| [`/v1/eval/ads-classifier`](#ads-classifier) | 广告关键字检测 |  json  | Json |



## pulp

> 图片剑皇服务用以检测图片的内容是否为正常(norm)、性感(sexy)或色情(pulp)<br>

Request

```
POST /v1/eval/pulp  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg" 
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
				"index": 2,	  
				"class": "norm", 
				"score": 0.9987
			}
		]
	}	
}
```
***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|index|{0,1,2}|类别编号, 即 0:pulp,1:sexy,2:norm|
|class|{"pulp","sexy","norm"}|图片内容鉴别结果，分为色情、性感或正常3类|
|score|float|将图片判别为某一类的准确度，取值范围0~1，1为准确度最高|

## pulp-small

> 图片剑皇服务用以检测图片的内容是否为正常(norm)、性感(sexy)或色情(pulp)<br>

Request

```
POST /v1/eval/pulp-small  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg" 
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
		"checkpoint":"endpoint",
		"confidences": [
			{
				"index": 2,	  
				"class": "norm", 
				"score": 0.9987
			},
			{
				"index": 0,	  
				"class": "pulp", 
				"score": 0.0012
			},
			{
				"index": 1,	  
				"class": "sexy", 
				"score": 0.0001
			}
		]
	}	
}
```
***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|checkpoint|{"endpoint","xxx"}|
checkpoint = endpoint(表示当前结果可直接给客户) 时，输出三类 normal,pulp,sexy ,其中normal的分数为实际分数，pulp和sexy是随机数
checkpoint = xxx(表示下一步需要调用的eval服务名称)时，输出三类，normal，pulp，sexy，其中normal和pulp的分数是真实的，sexy设为0|
|index|{0,1,2}|类别编号, 即 0:pulp,1:sexy,2:norm|
|class|{"pulp","sexy","norm"}|图片内容鉴别结果，分为色情、性感或正常3类|
|score|float|将图片判别为某一类的准确度，取值范围0~1，1为准确度最高|


## pulp-detect

> 图片剑皇服务用以检测图片的内容是否为正常(norm)、性感(sexy)或色情(pulp)<br>

Request

```
POST /v1/eval/pulp-detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg" 
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
				"index": 1,	  
				"class": "guns",
				"score": 0.997133,
				"pts": [[225,195], [351,195], [351,389], [225,389]]
			},
			...
		]
	}	
}
```
***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|index|{0～8}|类别编号|
|class|{"penis", "vulva", "sex","tits","breasts","nipples","ass","tback","anus"}|图片敏感区域鉴别结果|
|score|float|将图片判别为某一类的准确度，取值范围0~1，1为准确度最高|
|pts|list|敏感区域坐标|

## terror-classify

>根据图片内容判断其中含有的暴恐类物体，如正常(\_\_background__)、 藏独旗帜(Tibetan Flags，即雪山狮子旗)、伊斯兰国旗帜(Islamic Flags)、枪(Gun)等6个。<br>

Request

```
POST /v1/eval/terror-classify  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg" 
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
				"index": 3,	  
				"class": "Guns", 
				"score": 0.897
			}
		]
	}	
}

```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|index|{0,1,2,3,4,5}|index为类别编号|
|class|{normal, bloodiness, bomb, march, beheaded, fight}|
|score|float|将图片判别为某一类的准确度|


## terror-detect

> 暴恐检测

Request

```
POST /v1/eval/terror-detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/ms_face_detection6.jpg" 
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
				"index": 1,	  
				"class": "guns",
				"score": 0.997133,
				"pts": [[225,195], [351,195], [351,389], [225,389]]
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
|index|int|暴恐类别的编号|
|class|string|暴恐类别名称{\_\_background__,isis flag,islamic flag,knives,guns,tibetan flag}|
|score|float|物体检测框的准确度，取值范围0~1，1为准确度最高|
|pts|四点坐标值|[左上，右上，右下，左下]四点坐标框定的暴恐物体|



## facex-age

> 根据人脸判断年龄在0-2、4-6、8-13、15-20、25-32、38-43、48-53和60-100中的哪一个范围内<br>

Request：

```
POST /v1/eval/facex-age  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
   "data": {
	   "uri": “http://7xlv47.com0.z0.glb.clouddn.com/faceage.jpg”,   // 资源文件
	   "attribute": {
		   "pts":[[225,195],[351,195],[351,389],[225,389]] 
	   } 
   }
}
```

Response：

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"confidences": [
			{
				"index": 0,
				"class": "0-2",
				"score": 0.9753437638282776
			}
		]
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|pts|list[4]|图片中的人脸坐标，每张图片每次只发送一个人脸坐标|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|index|{0,1,2,4,5,6,7}|index为class对应的编号|
|class|{0-2,4-6,8-13,15-20,25-32,38-43,48-53,60-100}|8个年龄区间|
|score|float|将图片中人脸判别为某一年龄区间的准确度|

## facex-gender

> 根据人脸判断性别是男(Male)还是女(Female)<br>

Request：

```
POST /v1/eval/facex-gender  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
   "data": {
		"uri": “http://7xlv47.com0.z0.glb.clouddn.com/faceage.jpg”,   // 资源文件
		"attribute": {
			"pts":[[225,195],[351,195],[351,389],[225,389]]
		 } 
   }
}
```

Response：

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"confidences": [
			{
				"index": 0,
				"class": "Male",
				"score": 0.8863510489463806
			}
		]
	}
}
```
***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|pts|list[4]|图片中的人脸坐标，每张图片每次只发送一个人脸坐标|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|index|{0,1}|index为性别编号|
|class|{Male,Female}|性别|
|score|float|将图片中人脸判别为某一性别的准确度|

## facex-detect

> 人脸检测

Request

```
POST /v1/eval/facex-detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/ms_face_detection6.jpg" 
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
				"index": 1,	  
				"class": "face",
				"score": 0.997133,
				"pts": [[225,195], [351,195], [351,389], [225,389]],
				"quality": "clear"
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
|index|{1}|index值始终为1|
|class|{"face"}|class值始终为face|
|score|float|人脸框的准确度，取值范围0~1，1为准确度最高|
|pts|四点坐标值|[左上，右上，右下，左下]四点坐标框定的脸部|
|quality|string|人脸质量评估，取值范围{"clear","small","blur","cover","pose"}，其中"clear":清晰高质量人脸；"small":人脸框长或宽小于48的人脸；"blur":模糊人脸；"cover":遮挡人脸；"pose":大姿态人脸|



## facex-pose

> 人脸姿态</br>
>根据人脸回归出68点坐标定位信息及对应的3D姿态,仅支持单张图像多人脸模式

Request

```
POST /v1/eval/facex-pose  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://p9s1ibz34.bkt.clouddn.com/landmark.jpg",
		"attribute": {
			"detections": [{
				"pts": [
					[58.000000, 7.000000],
					[569.000000, 7.000000],
					[569.000000, 470.000000],
					[58.000000, 470.000000]
				]
			}]
		}
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|detections|array|通过facex-detect服务得到的图片中的人脸框集合|

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"landmarks": [{
			"landmark": [
				[69.32315063476563, 88.61177825927735],
				[65.63580322265625, 154.62063598632813],
				...
				[324.6520690917969, 351.4128112792969],
				[303.49444580078127, 347.1940612792969]
			],
			"pos": [-6.811560153961182, -6.722523212432861, 4.156128406524658]
		}]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|landmarks|array|人脸对应的检测结果集合|
|landmark|array|68个定位关键点|
|pos|array|3D姿态|

## detection

> 通用物体检测

Request

```
POST /v1/eval/detection  Http/1.1
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
				"score": 0.9842000007629395,
				"label_cn": "xx"
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
|label_cn|string|中文标签|


## object-classify

> 通用物体识别

Request

```
POST /v1/eval/object-classify  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg"
	},
	"params": {
		"limit": <limit>,
		"threshold": <threshold>
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|params.limit|int|可选参数。返回类别数，`limit=0`表示返回所有|
|params.threshold|float|可选参数。最小阈值，大于该阈值才能返回|


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
				"score": 0.3064107298851013,
				"label_cn": "xx"
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
|index|uint|识别分类序号|
|class|string|识别分类名称|
|score|float|识别的准确度，取值范围0~1，1为准确度最高|
|label_cn|string|中文标签|


## scene

> 通用场景识别
> 场景识别批量处理图片可以提高效率

Request

```
POST /v1/eval/scene  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

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
				"score": 0.3064107298851013,
				"label_cn": "xx"
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
|label_cn|string|中文标签|


## facex-feature
> 人脸特征提取<br>

Request：

```
POST /v1/eval/facex-feature  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
   "data": {
		 "uri": “http://oayjpradp.bkt.clouddn.com/age_gender_test.png”,   // 资源文件
		 "attribute": {
			"pts":[[23,343],[23,434],[323,434],[323,343]]
		 } 
   }
}
```

Response：

```
200 ok
Content-Type:application/octet-stream

stream
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|pts|list[4]|图片中的人脸坐标，每张图片每次只发送一个人脸坐标；可选，不填则获取全图特征|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|stream|float32 binary|返回由数值组成的二进制流,解析出来是[float32]的特征值列表|

## segment
> 物体分割<br>

Request：

```
POST /v1/eval/segment  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
   "data": {
		 "uri": <uri:string>   // 资源文件
   }
}
```

Response：

```
200 ok

{
	"code": 0,
	"message": "...",
	"result": {
		"mask": "data:image/png;base64,...",
		"detections": [
			{
				"index": <index:int>,
				"class": <class:string>,
				"color": [uint, uint, uint],
				"pts": [[<uint>, <uint>], [<uint>, <uint>], ...],
				"score": <score:float>
			},
			...
		]
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|mask|string|base64编码的分割染色图|
|index|uint|物体分割类别序号|
|class|string|物体分割类别名称|
|color|RGB|物体分割染色块颜色RGB|
|pts|四点坐标值|[左上，右上，右下，左下]四点坐标框定的物体|
|score|float|物体检测的准确度，取值范围0~1，1为准确度最高|


## facex-cluster

> 人脸聚类

Request

```
POST /v1/eval/facex-cluster  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
  "data": [
			  {															// group接口
				  "uri": "data:application/octet-stream;base64,xxx",	   //feature 值编码
				  "attribute":{
					 "cluster_id": -1,
					 "gt_id": -1
				  },
			  },
			  {												   
				  "uri": "data:application/octet-stream;base64,xxx",	   
				  "attribute":{
					  "cluster_id": 2,
					  "gt_id": 0
			  }
	]
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|base64 binary|Data URI Scheme形态的base64二进制人脸特征值数据|

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		 "facex_cluster": [
			  {  "cluster_id": -1, "cluster_center_dist": 0.0 },
			  {  "cluster_id": 0, "cluster_center_dist": 0.013042 }
		 ]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|cluster_id|int|新的类别号|
|cluster\_center_dist|float|与所属类别聚类中心之间的距离|


## politician

> 政治人物检索

Request

```
POST /v1/eval/politician  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "data:application/octet-stream;base64,xxx"
		"attribute":{
			"pts":[[23,343],[23,434],[323,434],[323,343]]
		}
	},
	"params":{
		"limit":2
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|Data URI形式的图片特征信息|
|limit|int|最大的返回相似人物个数|
|pts|list|人脸框，内部要根据人脸框大小来判断与哪个库对比|

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"confidences":[
			{
				"index": 1,
				"class": "xx",
				"group": "Inferior Artist",
				"score": 0.988,
				"sample": {
					"url": "",
					"pts": [[1213,400],[205,400],[205,535],[1213,535]]
				}
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
|confidences|list|结果列表，若请求中的pts算得最短边小于32则返回空list,若最短边小于60则与小脸库对比，否则与大脸库对比|
|index|uint|人物编号|
|class|string|人物姓名|
|group|string|人物分组信息，总共有7个组{'Domestic politician','Foreign politician',
'Sacked Officer(Government)','Sacked Officer (Enterprise)',
'Anti-China Molecule','Terrorist','Inferior Artist'}|
|sample.url|string|该政治人物的示例图片|
|sample.pts|list[4]|人脸在示例图片中的边框|


## search-bjrun

> bjrun特定图片库检索

Request

```
POST /v1/eval/search-bjrun  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "data:application/octet-stream;base64,xxx"
	},
	"params": {
		"limit": 1
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|Data URI形式的图片特征信息|
|limit|int|获取检索结果条目数|

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": [
		{
			"class": "xx",
			"score": 0.988
		},
		...
	]
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|class|string|示例图片地址|
|score|float|0~1,检索结果的可信度，1为确定|


## ocr-idcard

> 身份证信息识别

Request

```
POST /v1/eval/ocr-idcard  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/age_gender_test.png"
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|身份证图片链接|

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"name": "田淼淼",
		"people": "汉",
		"sex": "女",
		"address": "陕西省高碑店市庄发镇",
		"id_number": "1356648999203243269"	
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|name|string|姓名|
|people|string|民族|
|sex|string|性别|
|address|string|地址信息|
|id_number|string|身份证号码|


## image-feature
> 图片特征提取<br>
> image-feature-v2 升级版本输出的feature为2048个float共8192 bytes<br>

Request：

```
POST /v1/eval/image-feature  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
   "data": {
		 "uri": "http://oayjpradp.bkt.clouddn.com/age_gender_test.png"   // 资源文件
   }
}
```

Response：

```
200 ok
Content-Type:application/octet-stream

stream
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|stream|float32 binary|返回由数值组成的二进制流,解析出来是[float32]的特征值列表|


## video-feature

> 视频分类帧特征提取(cs, 私有部署)

Request

```
POST /v1/eval/video-feature  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/age_gender_test.png"
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片地址或base64编码的图片数据data:application/octet-stream;base64,xxx|

Response

```
200 ok

binary data
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
| |binary data|返回二进制数据，为2048个float32类型的feature值打包而成|


## video-classify 

> 视频分类(cs, 私有部署)

Request

```
POST /v1/eval/video-classify  Http/1.1
Content-Type:application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": [
		{
		"uri": "data:application/octet-stream;base64,xxxx"
		},
		{
		"uri": "data:application/octet-stream;base64,xxxx"
		},
		...
	]
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|data|uri list|一组图片的二进制feature数据，其中list 长度表示图片数目|
|uri|string|base64编码的二进制feature数据，每张图片的feature为2048个float32值|

Response

```
200 ok

{
  "code": 0,
  "message": "",
  "result": [
	{"9": 0.795},
	{"1": 0.082},
	{"2": 0.029},
	{"6": 0.027},
	{"7": 0.018}
  ]
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|result|dict{string:float32}|top5的分类结果{label:score}，按照score排序|



## ocr-classify 

> OCR图片类型分类

Request

```
POST /v1/eval/ocr-classify  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oayjpradp.bkt.clouddn.com/age_gender_test.png"
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
		"confidences": [{
			"index": 1,
			"class": "wechat",
			"score": 0.9887
		}]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|class|string|文本类别，{"wechat","blog","other-text","normal"}，分别表示微信、微博、其他文本、非文本|
|index|int|分类子项，[-1,32]|
|score|float|文本分类的置信度，0~1|




## ocr-detect 

> 微博微信场景OCR图片文字检测

Request

```
POST /v1/eval/ocr-detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
	},
	"params": {
		"image_type":"wechat"
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|ImageType|string|图片类型|

Response

```
200 ok

{
  "code": 0,
  "message": "",
  "result": {
	"area_ratio": 0.13991404297851073,
	"bboxes": [
		[140,1125,596,1161],
		[140,1046,331,1082],
		[140,1005,563,1041],
		[141,1167,238,1200],
		[140,924,594,962],
		[141,237,605,273],
		[140,965,365,999],
		[141,279,501,313],
		[148,785,449,821],
		[121,362,193,389],
		[139,558,303,592],
		[88,868,350,899],
		[139,417,302,454],
		[120,730,194,757],
		[628,1256,743,1305],
		[119,182,194,210],
		[119,502,194,531]
	],
	"img_type": "wechat"
  }
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|area_ratio|float|文本区域占全图的比例，0~1|
|bboxes|[[两点坐标]]|图片中所有的文本矩形框位置|
|image_type|string|文本分类类型|



## ocr-recognize 

> OCR图片文件识别

Request

```
POST /v1/eval/ocr-recognize  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
	},
	"params": {
		"image_type":"wechat",
		"bboxes": [
			[140,1125,596,1161],
			[140,1046,331,1082],
			...
			[119,182,194,210],
			[119,502,194,531]
		]
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|image_type|string|文本分类类型|
|bboxes|[][4]int|图片中所有的文本矩形框位置,[左上角x, 左上角y, 右上角x, 右上角y]|

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
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

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|bboxes|[][4][2]int|排列后的文本框的4点坐标|
|testx|[]string|对应文本框内的识别文字|




## ocr-scene-detect 

> 通用场景OCR图片文字检测 - EAST模型

Request

```
POST /v1/eval/ocr-scene-detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
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
	"bboxes": [
		[140,1125,596,1161,140,1046,331,1082],
		[140,1005,563,1041,141,1167,238,1200],
		[140,924,594,962,141,237,605,273]
	]
  }
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|bboxes|[[四点坐标]]|图片中所有的文本框位置，为顺时针方向旋转的任意四边形[x1,y1,x2,y2,x3,y3,x4,y4]|



## ocr-scene-recog

> 通用场景OCR图片文字识别

Request

```
POST /v1/eval/ocr-scene-recog  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
	},
	"params": {
		"bboxes": [
			[140,1125,596,1161,140,1046,331,1082],
			...
			[119,182,194,210,119,502,194,531]
		]
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|bboxes|[[四点坐标]]|图片中所有的文本框位置，为顺时针方向旋转的任意四边形[x1,y1,x2,y2,x3,y3,x4,y4]|

Response

```
200 ok

{
	"code": 0,
	"message": "",
	"result": {
		"texts": [
			{
				"bboxes": [36,388,594,395,593,439,35,433],
				"text": "7月部天我么"
			},
			...,
			{
				"bboxes": [119,182,194,210,119,502,194,531],
				"text": "高家小蓝单车网来信"
			},
		]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|bboxes|[[四点坐标]]|图片中所有的文本框位置，为顺时针方向旋转的任意四边形[x1,y1,x2,y2,x3,y3,x4,y4]|
|text|string|对应四边形框中的文字|



## police-detect

> 涉警检测

Request

```
POST /v1/eval/police-detect  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
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
	"code":0,
	"message":"",
	"result":{
		"detections":[
			{
				"class":"police_badge",
				"index":1,
				"pts":[[310,16],[352,16],[352,50],[310,50]],
				"score":0.9960552453994751
			},{
				"class":"police_uniform",
				"index":4,
				"pts":[[13,177],[497,177],[497,367],[13,367]],
				"score":0.9966244697570801
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
|class|类别|目前分为以下四类：police_badge, police_car_moto, police_car_vehicle, police_uniform|
|index|类别码|对应类别的编码：1:police_badge, 2:police_car_moto, 3:police_car_vehicle, 4:police_uniform|
|pts|检测坐标|检测所得对象在图中的四点坐标|
|score|置信度|0-1之前的数，数值越高，检测结果的准确度越高|


## ocr-sari-crann

> 身份证专用识别模型(高研院)

Request

```
POST /v1/eval/ocr-sari-crann  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://p9zv90cqq.bkt.clouddn.com/001.jpg"
	}，
	"params": {
		"bboxes": [
			[140,1125,596,1161,140,1046,331,1082],
			...
			[119,182,194,210,119,502,194,531]
		]
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|bboxes|[[四点坐标]]|图片中所有的文本框位置，为顺时针方向旋转的任意四边形[x1,y1,x2,y2,x3,y3,x4,y4]，或[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|

Response

```
200 ok

{
	"code":0,
	"message":"",
	"result":{
		"text":["河南省项城市芙蓉巷东四", "胡同2号", "性别‘女人民‘族汉", "412702199705127504", "1997年5月12日", "张杰"]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|text|识别文本|对应输入框的顺序，返回一个文本字符数组|


## ocr-sari-id-pre

> 身份证专用模型检测、识别预处理及信息机构化(高研院)

Request

```
POST /v1/eval/ocr-sari-id-pre  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://p9zv90cqq.bkt.clouddn.com/001.jpg"
	}，
	"params": {
		"type": "predetect",
		"bboxes": [
			[140,1125,596,1161,140,1046,331,1082],
			...
			[119,182,194,210,119,502,194,531]
		],
		"class": 0,
		"texts": ["河南省项城市芙蓉巷东四", "胡同2号", "性别‘女人民‘族汉", "412702199705127504", "1997年5月12日", "张杰"],
		"names": ["住址1", "住址3", "住址2", "性民", "公民身份号码", "出生", "姓名"],
		"regions": [
			[[120,225],[120,270],[440,270],[440,225]],
			[[120,305],[120,350],[440,350],[440,305]],
			...
			[[135,50],[135,100],[212,100],[212,50]]
		],
		"detectedBoxes": [
			[[121,231],[413,229],[413,260],[121,263]],
			[[72,173],[346,176],[345,205],[72,202]],
			...
			[[44,232],[119,233],[118,259],[44,258]]
		]
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|type|string|设定本次预处理的类型，"predetect":检测预处理，包括身份证正反面分类、身份证区域截图等；"prerecog":识别预处理，根据模版对齐、融合检测框；"postprocess":识别后处理，将各种文字信息根据字段配成对|
|bboxes|[[四点坐标]]|图片中所有的文本框位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|class|int|身份证正反面信息，0:正面；1:背面|
|texts|string|根据文字框数组，对应识别的文字字符数组|
|names|string|字段名称数组，由模版提供|
|regions|[[四点坐标]]|为模版设定的每个字段的大致位置坐标，[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|detectedBoxes|[[四点坐标]]|为检测算法检测出的文字字符框，[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|


Response

```
200 ok

{
	"code":0,
	"message":"",
	"result":{
		"class": 0,
		"alignedImg": "data:application/octet-stream;base64, ...",
		"names": ["住址1", "住址3", "住址2", "性民", "公民身份号码", "出生", "姓名"],
		"regions": [
			[[120,225],[120,270],[440,270],[440,225]],
			[[120,305],[120,350],[440,350],[440,305]],
			...
			[[135,50],[135,100],[212,100],[212,50]]
		],
		"detectedBoxes": [
			[[121,231],[413,229],[413,260],[121,263]],
			[[72,173],[346,176],[345,205],[72,202]],
			...
			[[44,232],[119,233],[118,259],[44,258]]
		],
		"bboxes": [
			[140,1125,596,1161,140,1046,331,1082],
			...
			[119,182,194,210,119,502,194,531]
		],
		"res": [
			["民族","汉"],
			["住址","河南省项城市芙蓉巷东四胡同2号"],
			["性别","女"],
			["公民身份号码","412702199705127504"],
			["出生","1997年5月12日"],
			["姓名","张杰"]
		]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|class|int|身份证正反面信息，0:正面；1:背面|
|alignedImg|string|图片截取身份证区域后的图片 base64 编码|
|names|string|字段名称数组，由模版提供|
|regions|[[四点坐标]]|为模版设定的每个字段的大致位置坐标，[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|detectedBoxes|[[四点坐标]]|为检测算法检测出的文字字符框，[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|bboxes|[[四点坐标]]|图片中所有的文本框位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|
|res|[字符串对]|识别后信息结构化的结果，["姓名","张杰"]|


***使用详细说明：***

根据请求type类型的不同，请求与返回的参数有不同的强制要求

|type|req非空参数|res相应返回的非空字段|
|:---|:---|:---|
|predetect|uri,type|alignedImg,names,regions,boxes|
|prerecog|uri,type,class,detectedBoxes,names,regions,boxes|boxes|
|postprocess|uri,type,class,boxes,texts,regions,names|res|

## ocr-sari-vat

> 增值税发票识别(高研院)

检测请求

Request

```
POST /v1/eval/ocr-sari-vat  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://pbqb5ctvq.bkt.clouddn.com/YBZZS_01488003.jpg"
	}，
	"params": {
		"type": "detect"
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|type|string|设定本次预处理的类型，"detect": 请求类型为文字检测|


Response

```
200 ok

{
	"code":0,
	"message":"",
	"result":{
		"bboxes": [
			[[140,1125],[596,1161],[140,1046],[331,1082]],
			...
			[[119,182],[194,210],[119,502],[194,531]]
		]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|bboxes|list[4,2]<float32>|图片中所有的文本框位置，为顺时针/逆时针方向旋转的任意四边形[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|


识别后处理请求

Request

```
POST /v1/eval/ocr-sari-vat  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://pbqb5ctvq.bkt.clouddn.com/YBZZS_01488003.jpg"
	}，
	"params": {
		"type": "postrecog",
		"texts": ["成都金睿通信有限公司","2016年6月27日","00212050","￥20256.41","5100153130","成都市高新区高新大道创业路14-16号","028-87492600","17%","17%","17%","","FC/APC-3.0","FC/PC-3.0","FC/APC-0.9","中国银行股份有限公司成都晋阳分理处125263950141","光纤跳线散件","光纤跳线散件","光纤跳线散件","贰万叁仟柒佰圆整","套","套","套","10000","30000","20000","0.3846153846","0.3418803419","0.3076923077","四川省绵阳科创园区园艺东街8号0816-2536680","653.85","1743.59","1046.15","9151010057461436xK","四川光发科技有限公司","3846.15","10256.41","6153.85","69","91510700MA62474D2N","皮广元","中国银行绵阳涪城支行123912372612","￥3443.59","￥23700.00","抵拉联","gU","皮广元","","18639+31/054*+4404>9523-796","68>2>0>5*4>08*14+2637-6-/","4//3<<+-*/+*2659*+8145<1530","+6147+/2998/548-495884>+2","00212060","5100153130","四川增值税专用发票"]
	}
}
```

***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|uri|string|图片资源地址|
|type|string|设定本次预处理的类型，"postrecog": 请求类型为文字结果后处理|
|texts|list[]<string>|待结构化的识别文字|


Response

```
200 ok

{
	"code":0,
	"message":"",
	"result":{
		"_BeiZhu":"20151023004",
		"_DaLeiMingCheng":"",
		"_DaiKaiBiaoShi":"",
		"_DaiKaiJiGuanDiZhiJiDianHua":"",
		"_DaiKaiJiGuanGaiZhang":"",
		"_DaiKaiJiGuanHaoMa":"",
		"_DaiKaiJiGuanMingCheng":"",
		"_DanZhengMingCheng":"四川增值税专用发票",
		"_FaPiaoDaiMa_DaYin":"5100152130",
		"_FaPiaoDaiMa_YinShua":"5100152130",
		"_FaPiaoHaoMa_DaYin":"00581608",
		"_FaPiaoHaoMa_YinShua":"00581508",
		"_FaPiaoJianZhiZhang":"",
		"_FaPiaoLianCi":"发票联",
		"_FaPiaoYinZhiPiHanJiYinZhiGongSi":"",
		"_FuHeRen":"",
		"_GaiZhangDanWeiMingCheng":"",
		"_GaiZhangDanWeiShuiHao":"",
		"_GouMaiFangDiZhiJiDianHua":"成都市成华区东三环二段龙潭工业园",
		"_GouMaiFangKaiHuHangJiZhangHao":"建行成都龙潭支行",
		"_GouMaiFangMingCheng":"成都博瑞特通信设备有限公司",
		"_GouMaiFangNaShuiShiBieHao":"510108590236573",
		"_HeJiJinE_BuHanShui":"￥3034.79",
		"_HeJiShuiE":"￥515.91",
		"_JiQiBianHao":"",
		"_JiaShuiHeJi_DaXie":"ⓧ叁仟伍佰伍拾圆零柒角整",
		"_JiaShuiHeJi_XiaoXie":"￥3550.70",
		"_JiaoYanMa":"",
		"_KaiPiaoRen":"杨婧",
		"_KaiPiaoRiQi":"2015年11月1日",
		"_MiMa":"70>3<6<6*0<8>25949552>77/854947*1803*/28+586<887-1--2/20/36388152-*2/<1+41752/77/85/062*0-170<-89586<<95-",
		"_ShouKuanRen":"",
		"_WanShuiPingZhengHao":"",
		"_XiaoLeiMingCheng":"一般增值税发票",
		"_XiaoShouDanWeiGaiZhangLeiXing":"",
		"_XiaoShouFangDiZhiJiDianHua":"四川省绵阳科创园区园艺东街8号 13990103663",
		"_XiaoShouFangKaiHuHangJiZhangHao":"中国银行绵阳涪城支行123912372612",
		"_XiaoShouFangMingCheng":"四川光发科技有限公司",
		"_XiaoShouFangNaShuiRenShiBieHao":"510790567644042",
		"_XiaoShouMingXi":[
			["光跳线SC/UPC-LC/UPC-SM3.0","5M","根","60","6.1538461538","369.23","17%","62.77"],
			["光跳线FC/UPC-LC/UPC-SM3.0","1OM","根","24","7.0085470085","168.21","17%","28.59"],
			["光跳线FC/UPC-LC/UPC-SM3.0","5M","根","52","6.1538461538","320.00","17%","54.40"],
			["尾纤FC/UPC-SM3.0","10M","根","420","4.2735042735","1794.87","17%","305.13"],
			["尾纤SC/UPC-SM3.0","10M","根","52","4.2735042735","222.22","17%","37.78"],
			["束状尾纤FC/UPC12芯","1.5M","根","3","32.905982906","98.72","17%","16.78"],
			["法兰盘FC/UPC","","颗","36","1.7094017094","61.54","17%","10.46"]
		]
	}
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|result|map<>|识别的结构化信息|



## ocr-ctpn 

> 长文本OCR图片文字检测 - CTPN模型

Request

```
POST /v1/eval/ocr-ctpn  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://oqgascup5.bkt.clouddn.com/ocr/WechatIMG61.png"
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
	"bboxes": [
		[[140,1125],[596,1161],[140,1046],[331,1082]],
		...
		[[119,182],[194,210],[119,502],[194,531]]
	]
  }
}
```

***返回字段说明：***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|bboxes|list[4,2]<int>|图片中所有的文本框4个点的坐标（顺时针），[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]|

## ads-detection

> 图片商业广告检测<br>

Request

```
POST /v1/eval/ads-detection  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg" 
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
		"pts": [
		[140,1125,596,1161,140,1046,331,1082],
		[140,1005,563,1041,141,1167,238,1200],
		[140,924,594,962,141,237,605,273]
	]
	}	
}
```
***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|data.uri|string|图片资源地址|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|pts|array|图片中所有文本框位置的四点坐标|

## ads-recognition

> 图片商业广告检测<br>

Request

```
POST /v1/eval/ads-recognition  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
	"data": {
		"uri": "http://cdn.duitang.com/uploads/item/201205/24/20120524122218_YR5Mz.jpeg" 
		"attribute": {
		   "pts": [
		    	[140,1125,596,1161,140,1046,331,1082],
			    ...
			    [119,182,194,210,119,502,194,531]
		    ]
        }
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
		"texts": [
			{
				"pts": [36,388,594,395,593,439,35,433],
				"text": "7月部天我么"
			},
			...,
			{
				"pts": [119,182,194,210,119,502,194,531],
				"text": "高家小蓝单车网来信"
			},
		]
	}	
}
```
***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|data.uri|string|图片资源地址|
|data.attribute.pts|list|ads-detection 接口返回的所有文本框位置|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|pts|array|图片中所有文本框位置的四点坐标|
|text|string|对应四边形框中的文字|


## ads-classifier

> 图片商业广告文本分类<br>

Request

```
POST /v1/eval/ads-classifier  Http/1.1
Content-Type: application/json
Authorization: Qiniu <AccessKey>:<Sign>

{
 "data":{
   "text":["系八十多分"，“AK47 是对冯绍峰”]
   },
  "params":{
     "type":[ "terror","ads"]
  }
}
```

Response

```
200 ok

{
  "code": "0",
  "message": "",
  "result": {
      "ads":{ 
        "summary":{
             "label": "ads",
             "score": 0.89
     	},
    	"confidences":[
    	{
         	"keys": [],
         	"label": "normal",
        	"score": 0.98
    	}，
    	{
         "keys": ["AK47"],
         "label": “ads”,
         "score": 0.98
    	}
        ]
	   },
  	   "terror":{
        	"summary":{},
        	"confidences":[],
	   }
	}
  }
}
```
***请求字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|text|list|待检测文本列表|
|type|list|文本检测类型包括"ads","terror","pulp","politician"|

***返回字段说明:***

|字段|取值|说明|
|:---|:---|:---|
|code|int|0:表示处理成功；不为0:表示出错|
|message|string|描述结果或出错信息|
|ads|obj|当请求type含有"ads"字段时返回广告检测ads的结果|
|summary|obj|当前type下对所有texts的最终综合判定|
|summary.label|string|当前type对应最终综合所有texts结果标签,取值{"normal",<"terror"|"pulp"|"ads"|"politician">}|
|confidences|list|texts中每一条文本在对应type下的判定结果|
|confidences.keys|list|对应type下单条文本中检测到的词库关键词列表|
|confidences.label|list|单条文本判定标签，取值{"normal",<"terror"|"pulp"|"ads"|"politician">}|