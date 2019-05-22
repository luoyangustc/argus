# ccp review 接口文档

- 测试环境地址：http://argus-job-gate.cs.cg.dora-internal.qiniu.io:5001
- 统一采用 JSON 格式，所以 content-type=application/json
- 接口错误统一返回： 599

## 接口详情

### 创建 Set

POST /v1/submit/bucket-censor

请求参数：

```
{
  uid: uint32,
  utype: uint32, // 用户类型, 普通用户为4, vip为2, 建议为4
  zone: int, // 区域, z0 华东
  bucket: string,
  prefix: string,
  mimetypes []string, // 可选, 默认为 ["vide", "image"]
  params: {
    image: {
      scenes: [ // 审核类型, pulp, terror, politician
        "pulp",
        "terror",
      ],
      params: { // 审核对应类型的参数, 默认为空
        pulp: {  // 审核对应类型的参数, 默认为空
        },
        terror: {  // 审核对应类型的参数, 默认为空
        },
      },
    },
    video: {
      scenes: [ // 审核类型
        "pulp",
        "terror",
      ],
      params: { // 审核对应类型的参数
        scenes: {
          pulp: {  // 审核对应类型的参数, 默认为空
          },
          terror: {   // 审核对应类型的参数, 默认为空
          },
        },
        save: { // 截帧结果保存地址
	      uid: uint32,
	      zone: int,
	      bucket: string,
	      prefix: string
        },
        vframe:  {
          mode: int  // 0表示自定义间隔，1表示关键帧
          interval: float64   // 单位秒
        },
        hookURL: string, // 运行完之后回调地址
      }
    },
  }
  save: { // 机审结果保存设置
    uid: uint32,
    zone: int,
    bucket: string,
    prefix: string
  }
}
```

成功返回：200
```
{
    job_id: string
}
```

### 回调

```
{
    keys: []string // 返回存储的bucket的key列表
}
```
