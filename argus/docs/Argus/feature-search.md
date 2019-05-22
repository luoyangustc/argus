<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1.特征集接口](#1%E7%89%B9%E5%BE%81%E9%9B%86%E6%8E%A5%E5%8F%A3)
  - [1.1. 创建特征集](#11-%E5%88%9B%E5%BB%BA%E7%89%B9%E5%BE%81%E9%9B%86)
  - [1.2. 销毁特征集](#12-%E9%94%80%E6%AF%81%E7%89%B9%E5%BE%81%E9%9B%86)
  - [1.3. 获取特征集信息](#13-%E8%8E%B7%E5%8F%96%E7%89%B9%E5%BE%81%E9%9B%86%E4%BF%A1%E6%81%AF)
  - [1.4. 更新特征集状态](#14-%E6%9B%B4%E6%96%B0%E7%89%B9%E5%BE%81%E9%9B%86%E7%8A%B6%E6%80%81)
- [2.特征接口](#2%E7%89%B9%E5%BE%81%E6%8E%A5%E5%8F%A3)
  - [2.1. 添加特征](#21-%E6%B7%BB%E5%8A%A0%E7%89%B9%E5%BE%81)
  - [2.2. 删除特征](#22-%E5%88%A0%E9%99%A4%E7%89%B9%E5%BE%81)
  - [2.3. 检索特征](#23-%E6%A3%80%E7%B4%A2%E7%89%B9%E5%BE%81)
  - [2.4. 查询特征](#24-%E6%9F%A5%E8%AF%A2%E7%89%B9%E5%BE%81)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1.特征集接口
## 1.1. 创建特征集
> 新建特征集，单次请求创建单个特征集

**Request**
```
Post /v1/sets/<set_id>
Content-Type: application/json

{
    "dimension": 512,
    "precision": 4,
    "size": 300000,
    "version": 0,
    "state": 1
}
```

**Response**
```
200 OK
```

接口字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|
|dimension|int|特征维度，选填，默认dimension=512|
|precision|int|特征精度，选填，默认precision=4，即float32|
|size|int|人脸库预期特征数目，必填|
|version|uint64|特征集版本号，单调递增，选填，默认version=0|
|state|int|特征集服务状态，{0,1,2}分别表示未知错误，已创建，已初始化，选填，默认state=0|

## 1.2. 销毁特征集
> 销毁特征集，单次请求删除单个特征集

**Request**
```
Post /v1/sets/<set_id>/destroy
```

**Response**
```
200 OK
```

接口字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|

## 1.3. 获取特征集信息
> 获取特征集信息，单次请求单个特征集

**Request**
```
Get /v1/sets/<set_id>
```

**Response**
```
200 OK
Content-Type: application/json

{
    "dimension": 512,
    "precision": 4,
    "version": 0,
    "state": 1
}
```

接口字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|
|dimension|int|特征维度，选填，默认dimension=512|
|precision|int|特征精度，选填，默认precision=4，即float32|
|version|uint64|特征集版本号，单调递增，选填，默认version=0|
|state|int|特征集服务状态，{0,1,2}分别表示未知错误，已创建，已初始化，选填，默认state=0|

## 1.4. 更新特征集状态
> 更新特征集状态，配合特征服务使用，控制状态，单次更新单个特征集状态

**Request**
```
Post /v1/sets/<set_id>/state/<set_state>
```

**Response**
```
200 OK
```

接口字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|
|state|int|特征集服务状态，{0,1,2}分别表示未知错误，已创建，已初始化|


# 2.特征接口
## 2.1. 添加特征
> 添加特征向量，单次请求添加N个特征

**Request**
```
Post /v1/sets/<set_id>/add
Content-Type: application/json

{
    "features":[
        {
            "id": "AAAO054X4wzh",
            "value": "<base64_value_string>"
        },
        ...
    ],
}
```

**Response**
```
200 OK
```

接口字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|
|id|string|特征唯一标识，必填，默认12字节的reqid码|
|value|string|特征值，必填，经过base64编码的特征值|

## 2.2. 删除特征
> 删除特征向量，单次请求删除N条

**Request**
```
Post /v1/sets/<set_id>/delete
Content-Type: application/json

{
    "ids":[
        "AAAO054X4wzh",
        "AAAHdaa3wwzh"
    ]
}
```

**Response**
```
200 OK
Content-Type: application/json

{
    "deleted": [
        "AAAO054X4wzh"
    ]
}

```

接口字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|
|ids|string数组|期望删除的特征ID，必填，默认12字节的reqid码|
|deleted|string数组|已删除的特征ID，部分删除或者未找到目标ID，均返回200|

## 2.3. 检索特征
> 搜索特征集，单次请求搜索N条特征

**Request**
```
Post /v1/sets/<set_id>/search
Content-Type: application/json

{
    "features":[
        <feature>,
        ...
    ],
    "limit": 5,
    "threshold":0.85
}
```

**Response**
```
200 OK
Content-Type: application/json

{
    "search_results": [
        {
            "results": [
                {
                    "id": "AAAO054X4wzh",
                    "score": 0.9249905
                },
                ...
            ]
        },
        ...
    ]
}

```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|
|feature|byte数组|特征值，必填，特征精度和维度必须和所属特征集一致，小端字节序|
|limit|int|返回TOP N的搜索结果，选填，默认N=1|
|search_results|object数组|每个object一一对应输入特征的搜索结果|
|results|object数组|TOP N的搜索结果，按score降序排列|
|id|string数组|特征ID，默认12字节的reqid码|
|score|float32|搜索结果的相似度|

## 2.4. 查询特征
> 查询在库特征值

**Request**

```
GET /v1/sets/<set_id>/features/<feature_id>
```

请求字段说明：

|字段|取值|说明|
|:---|:---|:---|
|set_id|string|特征集唯一标识|
|feature_id|string|特征唯一标识|

**Response**

```
200 OK
Content-Type: application/json

{
    "value": <base64_feature_value>
}
```

返回字段说明：

|字段|取值|说明|
|:---|:---|:---|
|value|string|特征值，经过base64编码的特征值|