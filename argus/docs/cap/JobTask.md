
## POST /v1/job/job
> 创建job

### 请求

```
POST /v1/job/job
{  
    "jobType": "batch",
    "labelMode": "mode_pulp",
    "notifyURL": "http://argus-bcp.cs.cg.dora-internal.qiniu.io:5001/v1/cap/jobs/*/done"
}
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `jobType` | string | Y | job类型(bath/??) |
| `labelMode` | string | Y | 打标类型(mode_pulp,mode_terror,mode_politician) |
| `notifyURL` | string | Y | 回调url |

### 返回

```
{
    "jobId": "1532316410"
}
```

## POST /v1/job/job/*/Tasks
> 创建指定job中的task

### 请求

```
POST /v1/job/job/*/Tasks
{
    "jobId":"1532084082",
    "tasks":[
        {
            "id":"201807201854_1",
            "uri":"http://oquqvdmso.bkt.clouddn.com/atflow-log-proxy/images/pulp-2018-03-07T19-04-17-jadoSKVNvL7ov2iTgJ_IWw=="
        },
        {
            "id":"201807201854_2",
            "uri":"http://oquqvdmso.bkt.clouddn.com/atflow-log-proxy/images/pulp-2018-03-07T16-26-42-zVvY6FiuXiXU0fvd92ISYA=="
        }
    ]
}
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `jobId` | string | Y | job id|
| `tasks` | object | Y | 任务列表|
| `task.id` | string | Y | taskid|
| `task.uri` | string | Y | task uri |

### 返回

```
{
}
```