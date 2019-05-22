<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [API](#api)
  - [get/json/file](#getjsonfile)
    - [参数说明](#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)
    - [结果说明](#%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E)
  - [get/politician/images](#getpoliticianimages)
    - [结果说明](#%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E-1)
  - [get/politician/images/<ImageName>](#getpoliticianimagesimagename)
    - [结果说明](#%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E-2)
  - [post/politician/images](#postpoliticianimages)
    - [请求参数](#%E8%AF%B7%E6%B1%82%E5%8F%82%E6%95%B0)
  - [post/politician/images/delete](#postpoliticianimagesdelete)
    - [请求参数](#%E8%AF%B7%E6%B1%82%E5%8F%82%E6%95%B0-1)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# API

| 入口 | PATH | 说明 |
| :--- | :--- | :--- |
| GetJsonFile | #get/json/file | 获取所有保存的政治人物的图片信息的json文件 |
| GetPoliticianImages | #get/politician/images | 获取所有保存的政治人物的图片信息 |
| GetpoliticianImages_ | #get/politician/images/<ImageName> | 获取单个政治人物的图片信息 |
| PostpoliticianImages | #post/politician/images | 新增政治人物的图片信息 |
| PostpoliticianImagesDelete | #post/politician/images/delete | 删除某个政治人物的图片信息 |

## get/json/file

*Request*
```
GET /json/file

{
	"url":<url>,
}
```

### 参数说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `url` | string | 保存生成的json文件的url地址 |

*Response*

```
200 OK
Content-Type: application/json

{
    "name":<name>
}
```
### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `name` | string | 生成的json文件名 |

## get/politician/images

*Request*
```
GET /politician/images
```

*Response*

```
200 OK
Content-Type: application/json

{
    "result": 
	[
		{
			"name":<name>,
			"uris":[],
		},
		...
	]
}
```
### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `result.name` | string | 政治人物的姓名 |
| `result.uris` | []string | 政治人物的图片信息 |

## get/politician/images/<ImageName>

*Request*
```
GET /politician/images/<ImageName>
```

*Response*

```
200 OK
Content-Type: application/json

{
    "uris":[],
}
```

### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `uris` | []string | 政治人物的图片信息 |

## post/politician/images

*Request*
```
POST /politician/images

{
    "data": 
	[
		{
			"name":<name>,
			"uris":[],
		},
		...
	]
}
```
### 请求参数

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `data.name` | string | 政治人物的姓名 |
| `data.uris` | []string | 政治人物的图片信息 |

*Response*

```
200 OK
```

## post/politician/images/delete
*Request*
```
POST /politician/images/delete

{
    "data": 
	[
		{
			"name":<name>,
			"uris":[],
		},
		...
	]
}
```
### 请求参数

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `data.name` | string | 政治人物的姓名 |
| `data.uris` | []string | 政治人物的图片信息 |

*Response*

```
200 OK
```