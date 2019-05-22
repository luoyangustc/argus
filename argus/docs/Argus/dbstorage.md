<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1. 人脸入库服务](#1-%E4%BA%BA%E8%84%B8%E5%85%A5%E5%BA%93%E6%9C%8D%E5%8A%A1)
  - [1.1 创建任务](#11-%E5%88%9B%E5%BB%BA%E4%BB%BB%E5%8A%A1)
  - [1.2 启动任务](#12-%E5%90%AF%E5%8A%A8%E4%BB%BB%E5%8A%A1)
  - [1.3 停止任务](#13-%E5%81%9C%E6%AD%A2%E4%BB%BB%E5%8A%A1)
  - [1.4 删除任务](#14-%E5%88%A0%E9%99%A4%E4%BB%BB%E5%8A%A1)
  - [1.5 获得任务](#15-%E8%8E%B7%E5%BE%97%E4%BB%BB%E5%8A%A1)
  - [1.6 查询任务信息](#16-%E6%9F%A5%E8%AF%A2%E4%BB%BB%E5%8A%A1%E4%BF%A1%E6%81%AF)
  - [1.7 查询任务日志](#17-%E6%9F%A5%E8%AF%A2%E4%BB%BB%E5%8A%A1%E6%97%A5%E5%BF%97)
  - [1.8 下载任务日志](#18-%E4%B8%8B%E8%BD%BD%E4%BB%BB%E5%8A%A1%E6%97%A5%E5%BF%97)
- [2. 人脸入库工具](#2-%E4%BA%BA%E8%84%B8%E5%85%A5%E5%BA%93%E5%B7%A5%E5%85%B7)
  - [2.1 依赖](#21-%E4%BE%9D%E8%B5%96)
  - [2.2 功能说明](#22-%E5%8A%9F%E8%83%BD%E8%AF%B4%E6%98%8E)
  - [2.3 实现细节](#23-%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82)
  - [2.4 字段录入](#24-%E5%AD%97%E6%AE%B5%E5%BD%95%E5%85%A5)
  - [2.5 使用说明](#25-%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

基本语义说明
* 资源表示方式（URI）。通过统一方式定位、获取资源（图片、二进制数据等）
	* HTTP， 网络资源，形如：`http://host/path`、`https://host/path`
	* Data，Data URI Scheme形态的二进制文件，形如：`data:application/octet-stream;base64,xxx`。

# 1. 人脸入库服务

## 1.1 创建任务

> 新建任务，单次请求创建单个任务。新建任务时需指定所属的group（即对哪个group进行人脸入库）   
> 还可指定该次任务的相关配置（可选）   
> 新建任务时还需传入一csv或json文件，文件内包含所有欲入库人脸的id，uri，tag，description

**Request**

```
POST /v1/face/task/new
Content-Type: multipart/form-data

form参数：
"group_name"
"reject_bad_face"
"mode"
"file"

```

请求字段说明：

| 字段              | 取值     | 说明                     |
| :--------------- | :------ | :----------------------- |
| group_name       | string  | 该任务欲入库的人脸库group名，必选|
| reject_bad_face  | string  | 是否过滤低质量人脸，false为不过滤，true为过滤（不入库），可选，默认为不过滤。低质量定义：模糊、遮挡、大姿态、人脸朝向不为上的人脸 |
| mode             | string  | 人脸选择策略，可以设置为 SINGLE（只允许图片里面出现单张人脸，否则入库失败） 或者 LARGEST（如果存在多张人脸，使用最大的人脸），不填默认 SINGLE。注：该参数只对公有云有效 |
| file             | string  | 表单提交的csv文件，必选。支持csv和json文件，文件扩展名必须为.csv或.json |


file文件说明：

| 文件类型  | 说明     |
| :-------| :------ |
| csv     | 文件每行内容为 **id,uri,tag,desc** ,其中id和uri必选，tag和desc可选。csv文件第一行开始即为内容（即不包含header）| 
| json    | 文件每行内容为 {"image":{"id":"id1","uri":"uri1","tag":"tag1", "desc":"desc1"}} ,其中id和uri必选，tag和desc可选。  |

**Response**

```
200 OK
Content-Type: application/json
{
   "id" : ""
}
```

返回字段说明：

| 字段 | 取值    | 说明 |
| :---| :----- | :----- |
| id  | string | 创建成功返回的任务id |



## 1.2 启动任务

> 启动入库任务，若任务处于已完成状态则返回错误

**Request**

```
POST /v1/face/task/<task_id>/start
```

请求字段说明：

| 字段    | 取值    | 说明 |
| :------ | :----- | :----- |
| task_id | string | 任务id |

**Response**

```
200 OK
```


## 1.3 停止任务

> 停止入库任务，若任务不处于启动状态则返回错误   
> 停止后若再次调用启动任务接口，则任务会从上一次中断点继续执行   
> 若服务意外停止，则重启服务时会将所有之前处于启动状态的任务置为停止状态

**Request**

```
POST /v1/face/task/<task_id>/stop
```

请求字段说明：

| 字段    | 取值    | 说明 |
| :------ | :----- | :----- |
| task_id | string | 任务id |

**Response**

```
200 OK
```


## 1.4 删除任务

> 当任务不处于运行状态时（防止误操作），删除任务，且不能恢复（之后也无法再继续启动）

**Request**

```
POST /v1/face/task/<task_id>/delete
```

请求字段说明：

| 字段    | 取值    | 说明 |
| :------ | :----- | :----- |
| task_id | string | 任务id |


**Response**

```
200 OK
```


## 1.5 获得任务

> 获得指定人脸库下所有任务的id，可指定任务指定状态的任务

**Request**

```
GET /v1/face/task/<group_name>/list?status=<status>
```

请求字段说明：

| 字段       | 取值   | 说明                |
| :-----     | :----- | :-----------------|
| group_name | string | 人脸库名称   |
| status     | string | 只查询指定状态的任务，可选。取值有:"Created","Pending","Running","Stopping","Stopped","Completed", 分别对应：已创建未执行，在队列中等待执行，正在执行中，正在停止中，已停止，已完成 |


**Response**

```
200 OK
Content-Type: application/json

{
    "tasks": [
        {
            "id": "AAAO054X4wzh",
            "status": "Running"
        }
        {
            "id": "AAAHdaa3wwzh",
            "status": "Completed"
        }
        ...
    ]
}

```

返回字段说明：

| 字段    | 取值   | 说明          |
| :----- | :----- | :------------|
| tasks  | list   | 符合查询条件的入库任务列表  |
| id     | string | 入库任务的id  |
| status | string | 入库任务的状态 |



## 1.6 查询任务信息

> 获得任务的相关信息，包括所属的人脸库名、配置信息、当前状态、该任务所需处理的总数、该任务当前已处理的数目

**Request**

```
GET /v1/face/task/<task_id>/detail
```

请求字段说明：

| 字段    | 取值    | 说明 |
| :------ | :----- | :----- |
| task_id | string | 任务id |

**Response**

```
200 OK
Content-Type: application/json
{
    "group_name" : "",
    "config":{
        "reject_bad_face": false,
        "mode": "SINGLE",
    },
    "total_count" : 1000,
    "handled_count": 1000,
    "success_count": 980,
    "fail_count": 20,
    "status": "Completed",
    "last_error": ""
}
```

返回字段说明：

| 字段                     | 取值    | 说明                     |
| :---------------------- | :----- | :----------------------- |
| group_name              | string | 任务所属的人脸库group名|
| config.reject_bad_face  | bool   | 是否过滤低质量人脸 |
| config.mode             | string | 配置的人脸选择策略 |
| total_count             | int    | 该任务一共需要处理的人脸数 |
| handled_count           | int    | 该任务当前已处理完的人脸数 |
| success_count           | int    | 当前已处理完的人脸数中成功入库的个数 |
| fail_count              | int    | 当前已处理完的人脸数中未入库的个数 |
| status                  | string | 该任务当前状态。取值有:"Created","Pending","Running","Stopping","Stopped","Completed", 分别对应：已创建未执行，在队列中等待执行，正在执行中，正在停止中，已停止，已完成 |
| last_error              | string | 当任务运行出错停止时，该字段返回造成停止的错误原因 |



## 1.7 查询任务日志

> 每当入库人脸出现错误时，系统会将错误记录到日志
> 可随时查询任务日志，无论任务是否已结束

**Request**

```
GET /v1/face/task/<task_id>/log?skip=<skip>&limit=<limit>
```

请求字段说明：

| 字段     | 取值    | 说明 |
| :------ | :----- | :----- |
| task_id | string | 任务id   |
| skip    | int    | 指定跳过的日志数，可选，默认为0，即不跳过 |
| limit   | int    | 指定返回的日志数目，范围为1-1000。可选，默认值为1000|

**Response**

```
200 OK
Content-Type: application/json
{
    "logs": [
        {
            "uri":"",
            "code":"",
            "message":""
        }
    ],
    "total_count": 100
}
```

返回字段说明：

| 字段         | 取值    | 说明          |
| :---------- | :----- | :------------ |
| logs        | list   | 错误日志列表   |
| uri         | string | 出错的图片uri  |
| code        | int    | 错误码        |
| message     | string | 错误信息       |
| total_count | int    | 该任务的总错误日志数，可用于分页 |


错误码说明：

| 错误码 | 说明                |
| :-----| :-----------------|
| 101   | 图片uri不存在，无法下载  |
| 102   | 图片无法打开  |
| 103   | 图片重复  |
| 104   | 图片id已存在  |
| 201   | 图片不包含人脸  |
| 202   | 图片包含多张人脸  |
| 203   | 人脸小于50*50  |
| 204   | 人脸姿态过大  |
| 205   | 人脸模糊  |
| 206   | 人脸有遮挡 |
| 207   | 人脸朝向不为上 |
| 301   | 系统错误  |
| 599   | 其他错误  |


## 1.8 下载任务日志

> 除了可以查询日志外，系统还提供下载日志，下载文件为csv文件。格式为：人脸图片uri，错误码，错误信息

**Request**

```
GET /v1/face/task/<task_id>/log/download
```

请求字段说明：

| 字段    | 取值    | 说明 |
| :------ | :----- | :----- |
| task_id | string | 任务id |

**Response**

```
200 OK
Content-Type: application/octet-stream
***文件内容***
```

# 2. 人脸入库工具

## 2.1 依赖

> 该工具依赖feature_group或feature_group_private的api接口（取决于任务配置）

## 2.2 功能说明

> 读取指定文件夹下所有图片（包括子文件夹下图片）或指定文件中的url列表，调用接口入库，同时实现：

1. 去除重复文件，即重复文件不上传
2. 支持中断续传
3. 列出错误文件及错误原因（ex：非图片文件，图片损坏，图片不包含人脸，etc）

## 2.3 实现细节

> 多线程入库，且处理多线程下的中断续传（场景为海量文件，非大容量文件，重点为多线程模式下记录上次中断点）

1. 按filepath.walk()默认的方式获取文件夹内所有图片，按文件名字典排序。或逐行读取url列表文件内的url信息
2. 将每个图片包装成job，扔给任务池，多线程执行
3. 每个线程记录当前处理图片的index
4. 如中断，下次执行时首先获得所有线程记录的index，从这些index开始续传并跳过已处理的index
5. 每次处理完图片，线程会将该图片sha1值记录到本地文件，便于去重（主要是中断情况下的去重）。每次程序开始也会读取该文件获得已处理图片的sha1（如果有）
6. 入库的图片来源有两个：指定文件夹里的所有图片或指定url列表文件中的所有图片url


## 2.4 字段录入

> 调用人脸入库接口时，传入id, uri, tag, desc四个字段。目前支持三种图片源

1. 图片来源为文件夹时：

    | 字段   | 说明            |
    | :---- | :-------------- |
    | id    | 图片在系统中的路径 |
    | uri   | 图片base64值 |
    | tag   | 空  |
    | desc  | 空 |

2. 图片来源为csv文件时（文件后缀名为.csv），默认为每行格式 id,url,tag,desc。例：faceId,http://somesite.com/test.jpg,faceTag,faceDescription

    | 字段   | 说明                     |
    | :---- | :--------                |
    | id    | csv文件中的第一列值         |
    | uri   | csv文件中的第二列值         |
    | tag   | csv文件中的第三列值，如果有  |
    | desc  | csv文件中的第四列值，如果有  |

3. 图片来源为json文件时（文件后缀名为.json），每行为json格式，例：{"image":{"id":"id1","uri":"http://localhost:8090/11.jpg","tag":"tag1", "desc":"desc1"}}

    | 字段   | 说明       |
    | :---- | :--------  |
    | id    | json中的id字段值  |
    | uri   | json中的uri字段值    |
    | tag   | json中的tag字段值，该字段可没有  |
    | desc  | json中的desc字段值，该字段可没有  |

## 2.5 使用说明

> 工具包括可执行文件dbstorage_tool和配置文件dbstorage_tool.conf

1.  将dbstorage_tool和dbstorage_tool.conf拷贝至linux主机的同一文件夹中
2.  在终端中进入该文件夹，执行./dbstorage_tool -f dbstorage_tool.conf
3.  工具会在当前文件夹下创建文件夹dbstorage_log，用以记录所有入库的进度
4.  工具会在dbstorage_log文件夹下创建以入库源文件（夹）路径命名的文件夹，在其中创建两个文件夹log和processlog（不要删除，中断续传需要用到）
5.  log文件夹下的count文件记录已处理图片数，error文件记录所有未成功上传的图片路径及原因
6.  dbstorage_tool.conf配置文件字段说明：

    | 字段                           | 类型      | 说明                                                                              |
    | :---------------------------- | :-------- | :--------------------------------------------------------------------------------|
    | image_folder_path             | string    | 所有待入库的图片的文件夹路径 |
    | image_list_file               | string    | 所有待入库的图片信息的的文件 |
    | load_image_from_folder        | bool      | 为true时则使用image_folder_path参数从文件夹入库，为false时则使用image_list_file参数从文件入库 |
    | thread_num                    | int       | 该任务起用的线程数 |
    | feature_group_service.host    | string    | 人脸1:N服务地址 |  
    | feature_group_service.timeout | int       | 人脸1:N服务超时时间，单位为秒，0为不超时 |  
    | serving_service.host          | string    | 原子服务地址 | 
    | serving_service.timeout       | int       | 原子服务超时时间，单位为秒，0为不超时 |  
    | group_name                    | string    | 入库的人脸库名称 |
    | is_private                    | bool      | 该服务是对接公有云还是私有化，false为公有云，true为私有化 |
    | task_config                   | object    | 图片入库的配置 |
    | task_config.reject_bad_face   | bool      | 是否过滤低质量人脸，false为不过滤，true为过滤（不入库），可选，默认为不过滤。低质量定义：模糊、遮挡、大姿态、人脸朝向不为上的人脸 |
    | task_config.mode              | string    | 人脸选择策略，可以设置为 SINGLE（只允许图片里面出现单张人脸，否则入库失败） 或者 LARGEST（如果存在多张人脸，使用最大的人脸），不填默认 SINGLE。注：该参数只对公有云有效 |
