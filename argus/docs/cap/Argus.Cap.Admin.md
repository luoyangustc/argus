<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Admin API]
  - [资源实体](#资源实体)
	- [LabelTitle](#LabelTitle)
  - [API](#api)
    - [Argus-Cap](#argus-cap)
      - [POST/v1/admin/label/modes](#post/v1/admin/label/modes)
      - [POST/v1/admin/label/modes/delete/_](#post/v1/admin/label/modes/delete/_)
      - [GET/v1/admin/label/modes](#get/v1/admin/label/modes)
      - [GET/v1/admin/label/modes/_](#get/v1/admin/label/modes/_)
      - [POST/v1/admin/groups](#post/v1/admin/groups)
      - [POST/v1/admin/groups/delete/_](#post/v1/admin/groups/delete/_)
      - [GET/v1/admin/groups](#get/v1/admin/groups)
      - [GET/v1/admin/groups/_](#get/v1/admin/groups/_)
      - [POST/v1/admin/auditors](#post/v1/admin/auditors)
      - [POST/v1/admin/auditors/delete/_](#post/v1/admin/auditors/delete/_)
      - [GET/v1/admin/auditors](#get/v1/admin/auditors)
      - [GET/v1/admin/auditors/_](#get/v1/admin/auditors/_)

# 资源实体
## LabelTitle
```
<LabelTitle.object>
{	
    title:<string>,
	selected:<bool>,
}
```

### 参数说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `title` | string | 结果类型，例如normal，sexy|
| `selected` | bool | 是否选择该结果 |

# API
## post/v1/admin/label/modes
> 以admin的身份登陆，创建一个新的label/mode记录

### 请求

```
POST /v1/admin/label/modes

{
    "name":"mode_test",
    "labelTypes":[
        "classify.pulp"
    ],
    "labels":{
        "pulp":[
            {
                "title":"normal",
                "selected":true
            },
            {
                "title":"sexy",
                "selected":false
            },
            {
                "title":"pulp",
                "selected":false
            }
        ]
    }
}
```

### 参数说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `name` | string | 标注类型的名字 |
| `labelType` | []string | Auditor可标注任务类型 |
| `labels` | map[string][]LabelTitle.object | 标注类型的结果 |

### 返回

```
200 OK
```

## post/v1/admin/label/modes/delete/_
> 以admin的身份登陆，删除一个label/mode记录

### 请求

```
POST /v1/admin/label/modes/delete/<mode_name>
```

### 返回

```
200 OK
```

## get/v1/admin/label/modes
> 以admin的身份登陆，获得所有label/modes的信息

### 请求

```
GET /v1/admin/label/modes
```

### 返回

```
[
	{
		"name":"mode_pulp",
		"labelTypes":[
			"classify.pulp"
		],
		"labels":{
			"pulp":[
				{
					"title":"normal",
					"selected":true
				},
				{
					"title":"sexy",
					"selected":false
				},
				{
					"title":"pulp",
					"selected":false
				}
			]
		}
	},
	...
]
```
### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `name` | string | 标注类型的名字 |
| `labelType` | []string | Auditor可标注任务类型 |
| `labels` | map[string][]LabelTitle.object | 标注类型的结果 |

## get/v1/admin/label/modes/_
> 以admin的身份登陆，获得一个label/mode的信息

### 请求

```
GET /v1/admin/label/modes/<mode_name>
```

### 返回

```
200 OK

{
    "name":"mode_test",
    "labelTypes":[
        "classify.pulp"
    ],
    "labels":{
        "pulp":[
            {
                "title":"normal",
                "selected":true
            },
            {
                "title":"sexy",
                "selected":false
            },
            {
                "title":"pulp",
                "selected":false
            }
        ]
    }
}
```

### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `name` | string | 标注类型的名字 |
| `labelType` | []string | Auditor可标注任务类型 |
| `labels` | map[string][]LabelTitle.object | 标注类型的结果 |

## post/v1/admin/groups
> 以admin的身份登陆，创建一个新的group记录

### 请求

```
POST /v1/admin/groups

{
    "groupId":"g_test",
    "mode":"mode_pulp",
    "realtimeLevel":"batch",
    "level":""
}
```

### 参数说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `groupId` | string | 用户组ID |
| `mode` | string | Auditor可标注任务类型名字，“batch”，“realtime” |
| `realtimeLevel` | string | 是否实时 |
| `level` | string | |

### 返回

```
200 OK
```

## post/v1/admin/groups/delete/_
> 以admin的身份登陆，删除一个groups记录

### 请求

```
POST /v1/admin/groups/delete/<groupId>
```

### 返回

```
200 OK
```

## get/v1/admin/groups

> 以admin的身份登陆，获得所有groups的信息

### 请求

```
GET /v1/admin/groups
```

### 返回

```
200 Ok

[
	{
		"groupId":"g0",
		"mode":"mode_pulp",
		"realtimeLevel":"batch",
		"level":""
	},
	...
]

```

### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `groupId` | string | 用户组ID |
| `mode` | string | 该用户组中Auditor可标注类型名 |
| `realtimeLevel` | string | Auditor是实时或非实时，“batch”，“realtime” |
| `level` | string | |

## get/v1/admin/groups/_

> 以admin的身份登陆，获得一个groups的信息

### 请求

```
GET /v1/admin/groups/<groupId>
```

### 返回

```
200 OK

{
    "groupId":"g_test",
    "mode":"mode_test",
    "realtimeLevel":"batch",
    "level":""
}
```

### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `groupId` | string | 用户组ID |
| `mode` | string | 该用户组中Auditor可标注类型名 |
| `realtimeLevel` | string | Auditor是实时或非实时，“batch”，“realtime” |
| `level` | string | |

## post/v1/admin/auditors
> 以admin的身份登陆，创建一个新的auditor记录

### 请求

```
POST /v1/admin/auditors

{
    "id":"aid_test",
    "valid": "valid",
    "curGroup": "g_test",
    "ableGroups": ["g_test", "g_test2"],
    "sandOkNum": 0,
    "sandAllNum": 0,
}
```

### 参数说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `id` | string | 用户ID |
| `valid` | string | 是否合法用户，“valid”合法，“invalid”非法 |
| `curGroup` | string | 当前所属的组 |
| `ableGroups` | array | 用户可以在的组 |
| `sandOkNum` | int | 沙子验证通过数 |
| `sandAllNum` | int | 沙子验证总数 |

### 返回

```
200 OK
```

## post/v1/admin/auditors/delete/_
> 以admin的身份登陆，删除一个auditor记录

### 请求

```
POST /v1/admin/auditors/delete/<auditorId>
```

### 返回

```
200 OK
```

## get/v1/admin/auditors
> 以admin的身份登陆，获得所有auditors的信息

### 请求

```
GET /v1/admin/auditors
```

### 返回

```
200 Ok

[
	{
		"id":"aid_test",
        "valid": "valid",
        "curGroup": "g_test",
        "ableGroups": ["g_test", "g_test2"],
        "sandOkNum": 0,
        "sandAllNum": 0,
	},
	...
]

```

### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `id` | string | 用户ID |
| `valid` | string | 是否合法用户，“valid”合法，“invalid”非法 |
| `curGroup` | string | 当前所属的组 |
| `ableGroups` | array | 用户可以在的组 |
| `sandOkNum` | int | 沙子验证通过数 |
| `sandAllNum` | int | 沙子验证总数 |

## get/v1/admin/auditors/_

> 以admin的身份登陆，获得一个auditor的信息

### 请求

```
GET /v1/admin/auditors/<auditorId>
```

### 返回

```
200 OK

{
    "id":"aid_test",
    "valid": "valid",
    "curGroup": "g_test",
    "ableGroups": ["g_test", "g_test2"],
    "sandOkNum": 0,
    "sandAllNum": 0,
}
```

### 结果说明

| 变量 | 类型 | 说明 |
| :--- | :--- | :---: | 
| `id` | string | 用户ID |
| `valid` | string | 是否合法用户，“valid”合法，“invalid”非法 |
| `curGroup` | string | 当前所属的组 |
| `ableGroups` | array | 用户可以在的组 |
| `sandOkNum` | int | 沙子验证通过数 |
| `sandAllNum` | int | 沙子验证总数 |