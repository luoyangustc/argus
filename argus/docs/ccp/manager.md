# ccp 接口文档

- 测试Host: http://argus-ccp.cs.cg.dora-internal.qiniu.io:5001
- 线上Host: http://argus-ccp.xs.cg.dora-internal.qiniu.io:5001

## 枚举定义

- SourceType (资源来源类型)

描述 | 值
---- | ---
KODO 资源 | KODO

- JobType (任务类型)

描述 | 值
---- | ---
增量类型 | STREAM
批量类型 | BATCH

- MimeType (资源类型)

描述 | 值
---- | ---
照片 | IMAGE
视频 | VIDEO

- Scene（检测类别）

描述 | 值
---- | ---
鉴黄 | pulp
鉴恐 | terror
鉴政 | politician

- Status (规则状态)

描述 | 值
---- | ---
开启 | ON
关闭/结束 | OFF

## 接口定义

### 创建规则

POST /v1/rules

请求参数: 

```
{
    rule_id: <string>,
    type: <JobType>,
    source_type: <SourceType>,
    source_id: <string>,
    source: {
        buckets: [{
            bucket: <string>,
            prefix: <string>,
        }],
    },
    action: {
        disable: <bool>,
        threshold: {"pulp":{"pulp":0.9,"sexy":0.8},"terror":0.6},
        pipeline: <string>,
        pfop_name: <string>,
        pfop_name_video: <string>,
    },
    notify_url: <string>,
    saver: {
        is_on: <bool>,
        bucket: <string>,
        prefix: <string>,
    },
    image: {
        is_on: <bool>,
        scenes: <[]string>,
    },
    video {
        is_on: <bool>,
        scenes: <[]string>,
    },
    automatic {
        is_on: <bool>,
        job_id: <string>,
        image {
            scene_params: <jsonmap>,//预留
        },
        video {
            params: <jsonmap>,
            scene_params: <jsonmap>,//预留
        },
    },
    manual {
        is_on: <bool>,
        job_id: <string>,
        image {
            scene_params: <jsonmap>,//预留
        },
        video {
            scene_params: <jsonmap>,//预留
        },
    },
    review {
        is_on: <bool>,
    },
}
```
- rule_id 为规则ID
- type 必须指定类型
- source_type 和 source_id 唯一确定资源范围
- source 中指明资源的具体属性，如bucket / prefix等
- action 中指明审核后的操作以及扩展信息，如禁用、队列名等，可空
- notify_url 为回调地址，可空
- saver 为保存结果的空间，可空
- image、video 至少有一个is_on = true
- automatic 为机审以及对应配置
- manual 为人审以及对应配置
- review 为复审以及对应配置

成功返回200，以及对应rule；

### 关闭规则

POST /v1/rules/close/<rule_id>

成功返回200

### 查询规则

#### 查询单条

GET /v1/rules/<rule_id>

成功返回200，以及对应rule；

#### 查询多条

GET /v1/rules

请求参数：

```
{
    type: <JobType>,
    status: <Status>,
    source_type: <SourceType>,
    source: `{
        buckets: [{
            bucket: <string>,
            prefix: <string>,
        }],
    }`,
    only_last: <bool>,
    mime_types: <[]string>,
}
```

成功返回200，以及对应rule列表；

#### 查询多条（按资源查询）

GET /v1/rules/source/<source_type>/<source_id>

请求参数：

```
{
    status: <Status>,
}
```

成功返回200，以及对应rule列表；



