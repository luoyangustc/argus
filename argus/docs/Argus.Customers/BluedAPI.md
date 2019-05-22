<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Blued Argus-util 融合接口](#blued-argus-util-%E8%9E%8D%E5%90%88%E6%8E%A5%E5%8F%A3)
  - [blued融合目标检测](#blued%E8%9E%8D%E5%90%88%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B)
- [Blued Atserving 原子API](#blued-atserving-%E5%8E%9F%E5%AD%90api)
  - [目标检测 blued-d](#%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-blued-d)
    - [类标译名文档](#%E7%B1%BB%E6%A0%87%E8%AF%91%E5%90%8D%E6%96%87%E6%A1%A3)
  - [人物检测 blued-c](#%E4%BA%BA%E7%89%A9%E6%A3%80%E6%B5%8B-blued-c)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Blued Argus-util 融合接口

### blued融合目标检测

> 由atserving blued-d和blued-c原子api融合而成的检测接口;
> 当blud-c返回值index为1且socore大于0.8时才会返回blued-d检测到的结果

Request

```
POST v1/blued/detection Http/1.1
Content-Type: application/json
Authentication: Qiniu <AccessKey>:<Sign>

{
	"data": {
	    "uri": "http://oo15c9v7b.bkt.clouddn.com/05.jpg"
	}
}
```


***请求字段说明:***

| 字段   | 取值     | 说明     |
| :--- | :----- | :----- |
| uri  | string | 图片资源地址 |

Response 

```
{
    "code": 0,
    "message": "",
    "result": {
        "detections": [
            {
                "area_ratio": 0.10949808,
                "class": "leg_hair",
                "index": 15,
                "score": 0.912,
                "pts": [[435,292],[735,292],[735,497],[435,497]]},
            
            {
                "area_ratio": 0.07600266,
                "class": "feet",
                "index": 28,
                "score": 0.9738,
                "pts": [[367, 150],[556,150],[556,376][367,376]]
            }
        ]
    }
}

```

***返回字段说明：***

| 字段         | 取值     | 说明                                       |
| :--------- | :----- | :--------------------------------------- |
| code       | int    | 0:表示处理成功；不为0:表示出错                        |
| message    | string | 描述结果或出错信息                                |
| index      | uint   | 类标索引(序号)                                 |
| area_ratio | float  | Bounding-box和全图的面积比                      |
| class      | string | 类名                                       |
| score      | float  | 检测结果置信度，取值范围0~1(1最高)                     |
| pts        | list   | 物体Bounding-box 左上，右上，右下，左下(顺时针)四个顶点坐标 $ (x_i,y_i) $ |


## Blued Atserving 原子API

### 目标检测 blued-d

> 目标物体检测及识别API (目前30类)

Request

```
POST v1/eval/blued-d  Http/1.1
Content-Type: application/json
Authentication: Qiniu <AccessKey>:<Sign>

{
	"data": {
	    "uri": "http://oo15c9v7b.bkt.clouddn.com/05.jpg"
	}
}
```

***请求字段说明:***

| 字段   | 取值     | 说明     |
| :--- | :----- | :----- |
| uri  | string | 图片资源地址 |


Response

```
200 ok

{
    "code": 0,
    "message": "",
    "result": {
        "detections": [
            {
                "area_ratio": 0.27127928424369,
                "class": "tattoo",
                "index": 8,
                "pts": [
                    [
                        269,
                        14
                    ],
                    [
                        499,
                        14
                    ],
                    [
                        499,
                        312
                    ],
                    [
                        269,
                        312
                    ]
                ],
                "score": 0.9973000288009644
            },
            {
                "area_ratio": 0.2568386453129388,
                "class": "tattoo",
                "index": 8,
                "pts": [
                    [
                        25,
                        29
                    ],
                    [
                        246,
                        29
                    ],
                    [
                        246,
                        322
                    ],
                    [
                        25,
                        322
                    ]
                ],
                "score": 0.9670000076293945
            }
        ]
    }
}
```

***返回字段说明：***

| 字段         | 取值     | 说明                                       |
| :--------- | :----- | :--------------------------------------- |
| code       | int    | 0:表示处理成功；不为0:表示出错                        |
| message    | string | 描述结果或出错信息                                |
| index      | uint   | 类标索引(序号)                                 |
| area_ratio | float  | Bounding-box和全图的面积比                      |
| class      | string | 类名                                       |
| score      | float  | 检测结果置信度，取值范围0~1(1最高)                     |
| pts        | list   | 物体Bounding-box 左上，右上，右下，左下(顺时针)四个顶点坐标 $ (x_i,y_i) $ |


#### 类标译名文档

> 后期有需求可以直接返回中文类标

| ID       | 特征       | 英文                  |
| -------- | -------- | ------------------- |
| 1-a-01   | 胡子       | beard               |
| 1-a-02   | 黑框眼镜     | black_frame_glasses |
| 1-a-03   | 警帽       | police_cap          |
| 1-a-04   | 太阳镜      | sun_glasses         |
| 1-a-05   | 耳钉       | stud_earrings       |
| 1-a-06   | 口罩       | mouth_mask          |
| 1-a-07   | 刘海       | bangs               |
| 1-b-01   | 纹身       | tattoo              |
| 1-b-02   | 衬衫       | shirt               |
| 1-b-03   | 西装上衣     | suit                |
| 1-b-04   | 领带       | tie                 |
| 1-b-05   | 皮带       | belt                |
| 1-b-06   | 牛仔裤      | jeans               |
| 1-b-07   | 短裤       | shorts              |
| 1-b-08   | 腿毛       | leg_hair            |
| 1-b-09   | 军装       | military_uniform    |
| 1-b-10   | 警服       | military_uniform    |
| 1-b-11   | 背心       | under_shirt         |
| 1-b-12   | 手套       | gloves              |
| 1-b-13   | 胸肌       | pecs                |
| 1-b-14   | 腹肌       | abdominal_muscles   |
| 1-b-15   | 小腿       | calf                |
| 1-b-16   | 三角裤      | briefs              |
| 1-b-17   | 平角裤      | boxers              |
| 1-b-18   | 臀部       | butt                |
| *1-b-19* | *热裤*     | *hot_pants*         |
| 1-c-01   | 皮鞋       | leather_shoes       |
| 1-c-02   | 黑袜       | black_socks         |
| 1-c-03   | 白袜       | white_socks         |
| 1-c-04   | 光脚       | feet                |
| *1-c-05* | *鞋(非皮鞋)* | *non_leather_shoes* |

### 人物检测 blued-c

> 判断图片中是否有人

Request

```
POST v1/eval/blued-c  Http/1.1
Content-Type: application/json
Authentication: Qiniu <AccessKey>:<Sign>

{
	"data": {
	    "uri": "http://oo15c9v7b.bkt.clouddn.com/05.jpg"
	}
}
```

***请求字段说明:***

| 字段   | 取值     | 说明     |
| :--- | :----- | :----- |
| uri  | string | 图片资源地址 |


Response

```
200 ok

{
    "code": 0,
    "message": "",
    "result": {
        "confidences": [
            {
                "class": "nonhuman",
                "index": 0,
                "score": 0.9849376082420349
            }
        ]
    }
}

```

***返回字段说明：***

| 字段         | 取值     | 说明                                       |
| :--------- | :----- | :--------------------------------------- |
| code       | int    | 0:表示处理成功；不为0:表示出错                        |
| message    | string | 描述结果或出错信息                                |
| index      | uint   | 类标索引(序号)                                 
| class      | string | 类名，目前只有两类{0:nonhuman,1:human}                                       |
| score      | float  | 检测结果置信度，取值范围0~1(1最高)                     
