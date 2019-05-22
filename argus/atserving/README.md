<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [App](#app)
  - [App Name](#app-name)
  - [App Version](#app-version)
- [Image](#image)
  - [Image Name](#image-name)
- [配置](#%E9%85%8D%E7%BD%AE)
  - [ETCD](#etcd)
    - [/sts/hosts](#stshosts)
    - [/nsq/producer](#nsqproducer)
    - [/nsq/consumer](#nsqconsumer)
    - [/worker](#worker)
    - [/model](#model)
    - [/app/metadata](#appmetadata)
    - [/app/release](#apprelease)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# App

| 关键信息 | 说明 | 来源 | 可选 |
| :--- | :--- | :--- | :--- |
| name | 应用名 | 用户指定 ||
| version | 版本号 | 用户指定 ||
| image | 使用镜像 | 用户选择 ||
| model | 模型输入 | 用户选择 ||
| meta | app基本配置信息 | [代码库](https://gitlab.qiniu.io/qbox/deploy-test/tree/master/playbook/ava-serving/apps) | 可选 |
| config | version对应配置 | [代码库](https://gitlab.qiniu.io/qbox/deploy-test/tree/master/playbook/ava-serving/apps) | 可选 |

## App Name
推荐命名规则`ava-foo`

> `ava-foo`为实例管理中的app名称，实例维护（dora2）以此为准; 
> 
> `foo`为Serving中的cmd名称，Serving中所有逻辑以此为准

## App Version
推荐命名规则`yyyymmdd-vxxx-i`

| 字段 | 说明 | 可选 |
| :--- | :--- | :--- |
| `yyyymmdd`| 发布日期 ||
| `vxxx` | 对应`image`的版本 ||
| `i`| 同样内容多次发布自增 | 可选 |

# Image
| 关键信息 | 说明 | 来源 |
| :--- | :--- | :--- |
| name | 镜像名 | 用户指定 |
| DockerFile | 编译规则 | [代码库](https://github.com/qbox/ava/tree/dev/docker/app) |
| code | 算法代码 | [代码库](https://github.com/qbox/ava/tree/dev/docker/scripts) | 

## Image Name
推荐命名规则`ava-caffe-[gpu|cpu]-classify-vyyyymmdd-i`

| 字段 | 说明 | 可选 |
| :--- | :--- | :--- |
| `caffe` | 网络框架 ||
| `[gpu|cpu]` | gpu、cpu专用 | 可选 |
| `classify` | 模型类型 ||
| `vyyyymmdd-i` | 生成日期，对应`image`的版本 ||

# 配置

## ETCD

```
/ava/serving
├── /sts/hosts              // /ava/serving/sts/hosts: sts地址
├── /nsq
│   ├── /producer           // /ava/serving/nsq/producer: nsq地址
│   └── /consumer           // /ava/serving/nsq/consumer: nsq地址
├── /worker                 // For 维护
│   ├── /default            // /ava/serving/worker/default: worker默认配置
│   ├── /app/hello1
│   │   ├── /default        // /ava/serving/worker/app/hello1/default: 指定app配置
│   │   ├── /release/v1     // /ava/serving/worker/app/hello1/release/v1: 指定版本配置
│   │   └── /release/v2
│   ├── /app/hello2
│   │   ├── /default
│   │   ├── /release/v1
│   │   └── /release/v2
│   └── /app/hello3
│       ├── /default
│       ├── /release/v1
│       └── /release/v2
├── /model
│   ├── /foo1
│   │   └── /default        // /ava/serving/model/foo1/default: 模型对应配置
│   └── /foo2
│       └── /default
└── /app                    // For 发布
    ├── /default
    │   └── /metadata       // /ava/serving/app/default/metadata: app默认基本配置
    ├── /metadata
    │   ├── /hello1         // /ava/serving/app/metadata/hello1: app基本配置
    │   ├── /hello2
    │   └── /hello3
    └── /release
        ├── /hello1
        │   ├── /v1         // /ava/serving/app/release/hello1/v1: 对应某一次模型发布
        │   └── /v2
        ├── /hello2
        │   ├── /v1
        │   └── /v2
        └── /hello3
            ├── /v1
            └── /v2
```

### /sts/hosts
```
[
  {
    "key": "1",
    "host": "127.0.0.1:5555"
  },
  ...
]
```
### /nsq/producer
```
[
  "127.0.0.1:4150",
  ... 
]
```
### /nsq/consumer
```
[
  {
    "addresses": [
      "127.0.0.1:4161"
    ]
  }
]
```
### /worker
```
{
  "delay4batch": 100000000,
  "max_concurrent": 20,
  "timeout": 10000000000,
  "kodo": {
    "rs_host": ""
  }
}
```
### /model
```
{
  "image": "xxxxx",
  "flavor": "xxx",
  "use_cpu": false
}
```
### /app/metadata
```
{
  "batch_size": 10,
  "public": true,
  "owner": {
    "uid": 123,
	  "ak": "",
    "sk": ""
  }
}
```
### /app/release
```
{
  "tar_file": "http://xx/xxx.tar",
  "batch_size": 1,
  "image_width": 224,
  "custom_files": {
    "xx": "http://xxx/xxx",
    ...
  },
  "custom_values": {
    "xx": 1,
    "yy": "",
    ...
  },
  "phase": "staging"
}
```