<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [API 基础模板](#api-%E5%9F%BA%E7%A1%80%E6%A8%A1%E6%9D%BF)
- [OPS](#ops)
  - [pulp](#pulp)
  - [terror](#terror)
  - [terror_complex](#terror_complex)
  - [terror_detect](#terror_detect)
  - [terror_classify](#terror_classify)
  - [face_group_search](#face_group_search)
  - [detection](#detection)
  - [politician](#politician)
  - [face_detect](#face_detect)
  - [image_label](#image_label)
  - [terror_classify_clip](#terror_classify_clip)
  - [face_group_search_private](#face_group_search_private)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# API 基础模板

*Request*

```
POST /v1/video/<video_id>
Host: argus.atlab.ai
Content-Type: application/json
Authorization:Qiniu <AccessKey>:<Sign>

{
    "data": {
        "uri": "http://xx"
    },
    "params": {
        "async": <async:bool>,
        "vframe": {
            "mode": <mode:int>,
            "interval": <interval:float>
        },
        "save": {
            "bucket": <bucket:string>,
            "zone": <zone:int>,
            "prefix": <prefix:string>
        },
        "hookURL": "http://yy.com/yyy"
    },
    "ops": [
        {
            "op": <op:string>,
            "hookURL": "http://yy.com/yyy",
            "hookURL_segment": "http://yy.com/yyy",
            "params": {
                "labels": [
                    {
                        "label": <label:string>,
                        "select": <select:int>,
                        "score": <score:float>
                    },
                    ...
                ],
                "terminate": {
                    "mode": <mode:int>,
                    "labels": {
                        <label>: <max:int>
                    }
                }
            }
        },
        ... 
    ]
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `video_id` | string | Y | 视频唯一标识，异步处理的返回结果中会带上该信息 |
| `data.uri` | string | Y | 视频流地址 |
| `params.async` | boolean | X | 是否异步处理，默认值为`False` |
| `params.vframe.mode` | int | X | 截帧处理逻辑，可选值为`[0, 1]`，默认值为`1` |
| `params.vframe.interval` | int | X | 截帧处理参数 |
| `params.save.bucket` | string | X | 保存截帧图片的Bucket |
| `params.save.zone` | int | X | 保存截帧图片的Zone |
| `params.save.prefix` | string | X | 截帧图片名称的前缀，图片名称的格式为`<prefix>/<video_id>/<offset>` （图片命名格式仅供参考，业务请不要依赖此命名格式）|
| `params.hookURL` | string | X | 所有op处理结束后的回调地址 |
| `ops.op` | string | Y | 执行的推理cmd |
| `ops.op.hookURL` | string | X | 该op处理结束后的回调地址 |
| `ops.op.hookURL_segment` | string | X | 片段回调地址 |
| `ops.op.params.labels.label` | string | X | 选择类别名，跟具体推理cmd有关 |
| `ops.op.params.labels.select` | int | X | 类别选择条件，`1`表示忽略不选；`2`表示只选该类别 |
| `ops.op.params.labels.score` | float | X | 类别选择的可信度参数，当select=`1`时表示忽略不选小于score的结果，当select=`2`时表示只选大于等于该值的结果 |
| `ops.op.params.terminate.mode` | int | X | 提前退出类型。`1`表示按帧计数；`2`表示按片段计数 |
| `ops.op.params.terminate.label.max` | int | X | 该类别的最大个数，达到该阈值则处理过程退出 |

> * `params.vframe`
> 	* `mode==0`: 每隔固定时间截一帧，由`vframe.interval`指定，单位`s`, 默认为`5.0s`
> 	* `mode==1`: 截关键帧, 默认为该模式

*Response*

*`When params.async == False`*

> 分析动作完成返回

```
HTTP/1.1 200 OK
Content-Type: application/json

{
    <op:string>: {
        "labels": [
            {
                "label": <label:string>,
                "score": <score:float>
            },
            ...
        ],
        "segments": [
            {
                "offset_begin": <offset:int>,
                "offset_end": <offset:int>,
                "labels": [
                    {
                        "label": <label:string>,
                        "score": <score:float>
                    },
                    ...
                ],
                "cuts": [
                    {
                        "offset": <offset:int>,
                        "uri": <uri:string>,
                        "result": {}
                    },
                    ...
                ]   
            },
            ...
        ]
    }
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `op` | string | Y | 推理cmd |
| `op.labels.label` | string | Y | 视频的判定结果 |
| `op.lables.score` | float | Y | 视频的判定结果可信度 |
| `op.segments.offset_begin` | int | Y | 片段起始的时间位置 |
| `op.segments.offset_end` | int | Y | 片段结束的时间位置 |
| `op.segments.labels.label` | string | Y | 片段的判定结果 |
| `op.segments.labels.score` | float | Y | 片段的判定结果可信度 |
| `op.segments.cuts.offset` | int | Y | 截帧的时间位置 |
| `op.segments.cuts.uri` | string | X | 截帧的保存路径 |
| `op.segments.cuts.result` | interface | Y | 截帧的对应的结果，内容跟具体推理相关 |

*`When params.async == true`*

> 立即返回

```
HTTP/1.1 200 OK
Content-Type: application/json

{
    "job": <job_id:string>
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `job_id` | string | Y | 视频分析任务唯一标识 |


# OPS

## pulp

> 剑皇

*Request*

```
{
    ...
    "ops": [
        {
            "op": "pulp",
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> | `0`, `1`, `2` | ARGUS - `/v1/pulp` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "mode" : 0,
            "interval": 3
        }
    },
    "ops": [
        {
            "op": "pulp",
            "params": {
                "labels": [
                    {
                        "label": "0"
                    }
                ]
            }
        }
    ]
}
```

```
{
    "pulp": {
        "labels": [
            {
                "label": "2",
                "score": 0.9996557
            }
        ],
        "segments": [
            {
                "cuts": [
                    {
                        "offset": 0,
                        "result": {
                            "label": 2,
                            "review": false,
                            "score": 0.9996557
                        }
                    },
                    ...
                ],
                "labels": [
                    {
                        "label": "2",
                        "score": 0.9996557
                    }
                ],
                "offset_begin": 0,
                "offset_end": 3000
            }
        ]
    }
}
```

## terror

> 暴恐识别

*Request*

```
{
    ...
    "ops": [
        {
            "op": "terror",
            "params": {
                "other": {
                    "detail": <bool>
                }
            }
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> | `0`, `1` | ARGUS - `/v1/terror` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "mode" : 0,
            "interval": 1
        }
    },
    "ops": [
        {
            "op": "terror",
            "params": {
                "other": {
                    "detail": true
                }
            }
        }
    ]
}
```

```
{
    "terror": {
        "labels": [
            {
                "label": "0",
                "score": 0.99999964
            }
        ],
        "segments": [
            {
                "cuts": [
                    {
                        "offset": 0,
                        "result": {
                            "label": 0,
                            "class": "xxx",
                            "score": 0.99999964,
                            "review": false
                        }
                    },
                    ...
                ],
                "labels": [
                    {
                        "label": "0",
                        "score": 0.99999964
                    }
                ],
                "offset_begin": 0,
                "offset_end": 4000
            }
        ]
    }
}
```

## terror_complex

> 暴恐融合识别

*Request*

```
{
    ...
    "ops": [
        {
            "op": "terror_complex",
            "params": {
                "other": {
                    "detail": <bool>
                }
            }
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> | `0`, `1` | ARGUS - `/v1/terror/complex` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "mode" : 0,
            "interval": 1
        }
    },
    "ops": [
        {
            "op": "terror",
            "params": {
                "other": {
                    "detail": true
                }
            }
        }
    ]
}
```

```
{
    "terror": {
        "labels": [
            {
                "label": "0",
                "score": 0.99999964
            }
        ],
        "segments": [
            {
                "cuts": [
                    {
                        "offset": 0,
                        "result": {
                            "label": 0,
                            "classes": [
                                {
                                    "class": "xxx",
                                    "score": 0.99999964
                                },
                                {

                                    "class": "yyy",
                                    "score": 0.98994326
                                }
                            "score": 0.99999964,
                            "review": false
                        }
                    },
                    ...
                ],
                "labels": [
                    {
                        "label": "0",
                        "score": 0.99999964
                    }
                ],
                "offset_begin": 0,
                "offset_end": 4000
            }
        ]
    }
}
```


## terror_detect

> 暴恐检测

*Request*

```
{
    ...
    "ops": [
        {
            "op": "terror_detect",
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> | `__background__`,`isis flag`,`islamic flag`,`knives`,`guns`,`tibetan flag` | SERVING - `/v1/eval/terror-detect` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "ops": [
        {
            "op": "terror_detect"
        }
    ]
}
```

```
{
    "terror_detect":{
        "labels":[
            {
                "label":"guns",
                "score":0.9995359
            }
        ],
        "segments":[
            {
                "offset_begin":0,
                "offset_end":0,
                "labels":[
                    {
                        "label":"guns",
                        "score":0.9995359
                    }
                ],
                "cuts":[
                    {
                        "offset":0,
                        "result":{
                            "detections":[
                                {
                                    "index":2,
                                    "class":"guns",
                                    "score":0.9995359,
                                    "pts":[
                                        [
                                            169,
                                            207
                                        ],
                                        ...
                                    ]
                                },
                               ...
                            ]
                        }
                    }
                ]
            }
        ]
    }
}
```


## terror_classify

> 鉴暴恐

*Request*

```
{
    ...
    "ops": [
        {
            "op": "terror_classify",
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> | `normal`,`bloodiness`,`bomb`,`march`,`beheaded`,`fight` | SERVING - `/v1/eval/terror-classify` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "mode": 0,
            "interval": 5
        }
    },
    "ops": [
        {
            "op": "terror_classify"
    ]
}
```

```
{
    "terror_classify": {
        "labels": [
            {
                "label": "normal",
                "score": 0.99999964
            }
        ],
        "segments": [
            {
                "cuts": [
                    {
                        "offset": 0,
                        "result": {
                            "confidences": [
                                {
                                    "class": "normal",
                                    "index": 0,
                                    "score": 0.99999964
                                }
                            ]
                        }
                    },
                    ...
                ],
                "labels": [
                    {
                        "label": "normal",
                        "score": 0.99999964
                    }
                ],
                "offset_begin": 0,
                "offset_end": 0
            }
        ]
    }
}
```


## face_group_search

> 人脸识别

*Request*

```
{
    ...
    "ops": [
        {
            "op": "face_group_search",
            "params": {
                "other": {
                    "group": <group_id>
                }
            }
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> | 人名 | ARGUS - `/v1/face/group/<gid>/search` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "interval": 5
        }
    },
    "ops": [
        {
            "op": "face_group_search",
            "params": {
                "other": {
                    "group": <group_id>
                }
            }
        }
    ]
}
```

```
{
    "face_group_search": {
        "labels": [
            {
                "label": "xx",
                "score": 0.99
            }
        ],
        "segments": [
            {
                "offset_begin": 0,
                "offset_end": 0,
                "labels": [
                    {
                        "label": "xx",
                        "score": 0.99
                    }
                ],
                "cuts": [
                    {
                        "offset": 30000,
                        "result": {
                            "detections": [
                                {
                                    "boundingBox":{
                                        "pts": [[1213,400],[205,400],[205,535],[1213,535]],
                                        "score":0.998
                                    },
                                    "value": {
                                        "id": "xxx",
                                        "name": "xx",
                                        "score":0.9998
                                    }
                                }
                            ]
                        }
                    },
                    ...
                ]
            }
        ]
    }
}
```

## detection

> 通用物体检测

*Request*

```
{
    ...
    "ops": [
        {
            "op": "detection",
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> | 物体分类名称 | SERVING - `/v1/eval/detection` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "ops": [
        {
            "op": "detection",
        }
    ]
}
```

```
{
    "detection": {
        "labels": [
            {
                "label": "person",
                "score": 0.8423
            }
        ],
        "segments": [
            {
                "cuts": [
                    {
                        "offset": 2000,
                        "result": {
                            "detections": [
                                {
                                    "class": "person",
                                    "index": 124,
                                    "pts": [
                                        [
                                            435,
                                            2
                                        ],
                                        ...
                                    ],
                                    "score": 0.8423
                                }
                            ]
                        }
                    }
                ],
                "labels": [
                    {
                        "label": "person",
                        "score": 0.8423
                    }
                ],
                "offset_begin": 0,
                "offset_end": 2000
            }
        ]
    }
}
```


## politician

> 政治人物搜索

*Request*

```
{
    ...
    "ops": [
        {
            "op": "politician",
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> |  人名 | ARGUS - `/v1/face/search/politician` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "mode": 0,
            "interval": 5
        }
    },
    "ops": [
        {
            "op": "politician"
        }
    ]
}
```

```
{
    "politician": {
        "labels": [
            {
                "label": "2",
                "score": 0.9996557
            }
        ],
        "segments": [
            {
                "cuts": [
                    {
                        "offset": 0,
                        "result": {
                            "detections": [
                                {
                                    "boundingBox":{
                                        "pts": [[1213,400],[205,400],[205,535],[1213,535]],
                                        "score":0.998
                                    },
                                    "value": {
                                        "name": "xx",
                                        "score":0.567,
                                        "review": True
                                    },
                                    "sample": {
                                        "url": "",
                                        "pts": [[1213,400],[205,400],[205,535],[1213,535]]
                                    }
                                },
                                ...
                            ]
                        }
                    },
                    ...
                ],
                "labels": [
                    {
                        "label": "2",
                        "score": 0.9996557
                    }
                ],
                "offset_begin": 0,
                "offset_end": 0
            }
        ]
    }
}
```


## face_detect

> 人脸检测及属性

*Request*

```
{
    ...
    "ops": [
        {
            "op": "face_detect",
            ...
        },
        ...
    ]
}
```

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "mode": 0,
            "interval": 5
        }
    },
    "ops": [
        {
            "op": "face_detect"
        }
    ]
}
```

```
{
    "face_detect": {
        "labels": [
            {
                "label": "",
                "score": 0.9996557
            }
        ],
        "segments": [
            {
                "cuts": [
                    {
                        "offset": 0,
                        "result": {
                      		"detections": [
                    			 {
                    				"boundingBox":{
                    					"pts": [[138,200],[305,200],[305,535],[138,535]],
                    					"score":0.9998
                    				},
                    				"age":{
                    					"value": 26.87,
                    					"score":0.9876
                    				},
                    				"gender":{
                    					"value":"Female",
                    					"score":0.9657
                    				}
                    			},
                                ...
                    		]
                        }
                    },
                    ...
                ],
                "labels": [
                    {
                        "label": "",
                        "score": 0.9996557
                    }
                ],
                "offset_begin": 0,
                "offset_end": 0
            }
        ]
    }
}
```

## image_label

> 图片打标

*Request*

```
{
    ...
    "ops": [
        {
            "op": "image_label",
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> |  图片打标 | ARGUS - `/v1/image/label` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "params": {
        "vframe": {
            "mode": 0,
            "interval": 5
        }
    },
    "ops": [
        {
            "op": "image_label"
        }
    ]
}
```

```
{
    "image_label":{
        "labels":[
            {
                "label":"junkyard",
                "score":0.83428067
            },
            ...
        ],
        "segments":[
            {
                "offset_begin":0,
                "offset_end":4000,
                "labels":[
                    {
                        "label":"junkyard",
                        "score":0.83428067
                    },
                    ...
                ],
                "cuts":[
                    {
                        "offset":0,
                        "result":{
                            "confidences":[
                                {
                                    "class":"junkyard",
                                    "score":0.83428067
                                },
                                ...
                            ]
                        }
                    }, 
                   ...
                ]
            }
        ]
    }
}                                    
```

## terror_classify_clip

> 视频分类

*Request*

```
{
    ...
    "ops": [
        {
            "op": "terror_classify_clip",
            ...
        },
        ...
    ]
}
```

> Labels | Result |
> | :--- | :--- |
> |  图片打标 | SERVING - `/v1/eval/vedio-classify` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
    "ops": [
        {
            "op": "terror_classify_clip"
        }
    ]
}
```

```
{
    "terror_classify_clip":{
        "segments":[
            {
                "offset_begin":0,
                "offset_end":9809,
                "labels":[
                    {
                        "label":"9",
                        "score":0.85689926
                    }
                ],
                "clips":[
                    {
                        "offset_begin":0,
                        "offset_end":3803,
                        "result":{
                                "1":0.05045907,"2":0.013268032,"3":0.011439596,"6":0.016477935,"9":0.8826332
                            },
                    },
                    ...
                ]
            },
            ...
        ]
    }
}
```

## face_group_search_private

> 私有化人脸1:N搜索

*Request*

```
{
    ...
    "ops": [
        {
			"op": "face_group_search_private",
			"params": {
				"other": {
					"group": "test",
					"threshold": 0.0,
					"limit": 1
				}
			}
        },
        ...
    ]
}
```

| 参数 | 类型 | 必选 | 说明 |
| :--- | :--- | :---: | :--- |
| `ops.op` | string | Y | 执行的推理cmd |
| `ops.op.hookURL_cut` | string | X | 截帧回调地址 |
| `ops.op.params.other.group` | string | Y | 人脸搜索库ID，唯一标识人脸库 |
| `ops.op.params.other.threshold` | float | N | 人脸搜索相似度阈值，即相似度高于该值才返回结果，默认为0 |
| `ops.op.params.other.limit` | int | N | 相似人脸项数目限制，默认返回全部相似人脸 |

> Labels | Result |
> | :--- | :--- |
> | 人名 | ARGUS - `/v1/face/groups/<group_id>/search` |

*Example*

```
{
    "data": {
        "uri": "http://foo.com/foo.mp4"
    },
	"ops": [
        {
            "op": "face_group_search_private",
            "params": {
                "other": {
                    "group": "test",
                    "threshold": 0.0,
                    "limit": 1
                }
            }
        }
    ]
}
```

```
{
    "face_group_search_private":{
        "segments":[
            {
                "offset_begin":0,
                "offset_end":9809,
                "labels":[
                    {
                        "label":"张三",
                        "score":0.85689926
                    }
                ],
                "cuts":[
                    {
                        "offset": 5008,
                        "result": {
                            "faces": [
                                {
                                    "bounding_box": {
                                        "pts": [[1226,343], [1271,343], [1271,391], [1226,391]],
                                        "score": 0.99122447
                                    },
                                    "faces": [
                                        {
                                            "id": "RaXQ73asC54e6c01B6PfIg==",
                                            "score": 0.85689926,
                                            "tag": "张三"
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                    ...
                ]
            },
            ...
        ]
    }
}
```
**face_group_search_private.segments.cuts.result返回参数说明**
 
| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `faces.bounding_box.pts`     | array | 人脸所在图片中的位置，四点坐标值 |[左上，右上，右下，左下] 四点坐标框定的脸部 |
| `faces.[].bounding_box.score`     | float | 人脸的检测置信度 |
| `faces.faces`     | object | 和此人脸相似的人脸的列表 |
| `faces.faces.id`     | string | 相似人脸ID |
| `faces.faces.score`     | float | 两个人脸的相似度 |
| `faces.faces.tag`     | string | 相似人脸Tag |
| `faces.faces.desc`     | string | 相似人脸描述 |


