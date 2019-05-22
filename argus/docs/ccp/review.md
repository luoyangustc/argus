# ccp review 接口文档

- 测试环境地址：http://argus-ccp-review.cs.cg.dora-internal.qiniu.io:5001
- 统一采用 JSON 格式，所以 content-type=application/json
- 接口错误统一返回： 599

### 枚举类型定义：

- SourceType (资源来源类型)

描述 | 值
---- | ---
KODO 资源 | KODO
APi 资源 |  API

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
直播流 | LIVE

- Suggestion (判断结果)

描述 | 值
---- | ---
通过 | PASS
不确定 |  REVIEW
确认违规 |  BLOCK
封禁 | DISABLED

- Scene（检测类别）

描述 | 值
---- | ---
鉴黄 | pulp
鉴恐 | terror
鉴政 | politician

## 接口详情

### 创建 Set

POST /v1/sets

请求参数：

```
{
  set_id: string,
  source_type: SourceType （参考枚举类型说明）,
  type: JobType （参考枚举类型说明）,
  automatic: bool,
  manual: bool,
  bucket: string,
  prefix: string,
  notify_url: string
}
```

- set_id 
- source_type 和 type 必须为指定类型
- automatic 和 manual 中 automatic 一定为 true（automatic 为 true，manual 为 false 表示机审，两者都为true 表表示机+人）
- 如果是七牛资源，bucket 不能为空,  prefix 可为空
- notify_url 可为空

成功返回：200

### 查询 Set

GET /v1/sets/:set_id

请求参数：

- set_id 为创建时候提交的 set_id

返回：

```
=> {
  set_id: string,
  source_type: SourceType （参考枚举类型说明）,
  type: JobType （参考枚举类型说明）,
  automatic: bool,
  manual: bool,
  bucket: string,
  prefix: string,
  notify_url: string
}
```

### 增量添加单个 entry

POST /v1/sets/:set_id/entry

请求参数：

```
{
  resource: []byte,
  uri_get: string,
  mimetype: MimeType （参考枚举类型说明）,
  original: {
    source: string,
    suggestion: Suggestion （参考枚举类型说明）,
    scenes: {
      pulp: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      },
      terror: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      },
      politician: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      }
    }
  },
  final: {
    suggestion: Suggestion （参考枚举类型说明）,
    scenes: {
      pulp: {
        suggestion: Suggestion,
        score: float32,
      },
      terror: {
        suggestion: Suggestion,
        score: float32,
      },
      politician: {
        suggestion: Suggestion,
        score: float32,
      }
    }
  },

  result: []byte,
  video_cuts: [{
    uri: string,
    offset: int64,
    original: {
      source: string,
      suggestion: Suggestion （参考枚举类型说明）,
      scenes: {
        pulp: {
          suggestion: Suggestion,
          labels: [{
              label: string,
              score: float32,
              group: string,
              pts: [][2]int,
          }]
        },
        terror: {
          suggestion: Suggestion,
          labels: [{
              label: string,
              score: float32,
              group: string,
              pts: [][2]int,
          }]
        },
        politician: {
          suggestion: Suggestion,
          labels: [{
              label: string,
              score: float32,
              group: string,
              pts: [][2]int,
          }]
        }
      }
    } 
  }],
}
```

成功返回：200

### 批量添加多个 entry

POST /v1/sets/:set_id/entries

请求：

```
{
  uid: uint32,
  bucket: string,
  keys: [string],
}
```

成功返回：200

### 修改单个 entry

POST /v1/sets/:set_id/entries/:entry_id

请求：

```
{
  suggestion: Suggestion （参考枚举类型说明）,
  scenes: {
    pulp: {
      suggestion: Suggestion,
      score: float32,
    },
    terror: {
      suggestion: Suggestion,
      score: float32,
    },
    politician: {
      suggestion: Suggestion,
      score: float32,
    }
  }
}
```
成功返回：200

###  查询 entries

POST /v1/fetch/entries

请求：

```
{
  set_id: string,

  source_type: SourceType （参考枚举类型说明）,
  type: JobType （参考枚举类型说明）,
  automatic: bool,
  manual: bool,
  bucket: string,
  prefix: string,

  original: {
    source: string,
    suggestion: Suggestion （参考枚举类型说明）,
    scenes: {
      pulp: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      },
      terror: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      },
      politician: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      }
    }
  },
  final: {
    suggestion: Suggestion （参考枚举类型说明）,
    scenes: {
      pulp: {
        suggestion: Suggestion,
        score: float32,
      },
      terror: {
        suggestion: Suggestion,
        score: float32,
      },
      politician: {
        suggestion: Suggestion,
        score: float32,
      }
    }
  },

  suggestion: Suggestion（参考枚举类型说明）,
  mimetype: MimeType（参考枚举类型说明）,

  scene: Scene（参考枚举类型说明）,
  min: float32,
  max: float32,

  start: int64,
  end: int64

  marker: string,
  limit: int
}
```

- set_id 是选填，如果前端能够确定是哪个 set_id，那么其他相关的 filter 参数可以不传
- 如果 set_id 为空，那么 source_type 和 type，不能为空
- scene 和 min, max 需要成对出现 
- start，end 表示时间范围
- marker，limit 来控制分页 (其中marker是entry_id, 可以取上一次返回值中最后的一个entry_id)

成功返回：

```
=> [{
  // entry 所有字段 +
  bucket: string,
  automatic: bool,
  manual: bool,
  cover_uri: string,
}]
```
entry 信息参考其创建提交数据结构，在此基础上添加了 `id` 字段。

###  查询 entries 统计信息

POST /v1/sets/counters

请求参数：

```
{
  sets: [set_id]
}
```

成功返回：

```
=> [{
  set_id: string,
  image_values: {
    pulp: int
    terror: int
    politician: int
  },
  left_image_values: {
    pulp: int
    terror: int
    politician: int
  },
  video_values: {
    pulp: int
    terror: int
    politician: int
  },
  left_video_values: {
    pulp: int
    terror: int
    politician: int
  }
}]
```

注意：
- image_values 和 left_image_values 表示图片总计和剩余带 review 统计信息
- video_values 和 left_video_values 表示视频总计和剩余带 review 统计信息

###  查询 entry cuts 信息

GET /v1/entries/:id/cuts?marker=xx&limit=int

返回：

```
=> [{
  id: string,
  uri: string,
  offset: int64，
  original: {
    source: string,
    suggestion: Suggestion （参考枚举类型说明）,
    scenes: {
      pulp: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      },
      terror: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      },
      politician: {
        suggestion: Suggestion,
        labels: [{
            label: string,
            score: float32,
            group: string,
            pts: [][2]int,
        }]
      }
    }
  } 
}]
```

###  查询 entry cuts 总数

GET /v1/entries/:id/cuts/count

返回：

```
{
  total: int
}
```


### entry 状态更新回调通知

POST /NotifyURL

请求参数：

```
{
  type: enums.SourceType,
  set_id: string,
  bucket: string,
  uri_get: string,
  from: enums.Suggestion,
  to: enums.Suggestion,
}
```

成功请求： 回调地址返回 2xx 。


### 重置 set 

POST /sets/:set_id/reset

请求参数：

```
{
  source_type: enums.JobType,
  job_type: enums.SourceType,
  type: string, (soft|hard|remove)
}
```

|参数名称|参数值|
|-|-|
|job_type|BATCH/STEAM|
|source_type|KODO/BATCH|
|type|soft(清空统计数据并重新触发重新统计)/hard(清空统计数据,并等待上游重新推送数据以触发统计)/remove(清空整个set)|

慎重操作:
线上最好不要hard清空增量的set, 也不要remove set, 除非你意识到你的行为


成功请求： 回调地址返回 2xx 。
