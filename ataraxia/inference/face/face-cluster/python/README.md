# Atlab Inference for Face-cluster

## 请求格式

```json
{
    "datas": [
        {   // 第一个人脸特征信息
            "uri": <string>,            // 与body字段可二选一，face feature二进制文件的uri，
                                        // 如body字段有效，则忽略改uri
            "attribute": {
                "face_id": <string>     // 人脸id
                "group_id": <int>       // 前一次的聚类标签, 该字段为空或者-2，表示未参与过前一次聚类；
                                        // =-1，表示前一次未聚成功；大于等于0，表示前一次的聚类编号
            }
            "body": <binary stream>     // 与uri字段可二选一，人脸特征，二进制流
        },
        {   // 第二个人脸特征信息
            "uri": <string>,            // 与body字段可二选一，face feature二进制文件的uri，
                                        // 如body字段有效，则忽略改uri
            "attribute": {
                "face_id": <string>     // 人脸id
                "group_id": <int>       // 前一次的聚类标签, 该字段为空或者-2，表示未参与过前一次聚类；
                                        // =-1，表示前一次未聚成功；大于等于0，表示前一次的聚类编号
            }
            "body": <binary stream>     // 与uri字段可二选一，人脸特征，二进制流
        },
        ...
    ],
    "params": {
        "sim_thresh": <float>,   //可选，聚类阈值(相似度，非阈值)参数，可在初始化config中设置，默认值：0.45
        "min_samples_per_cluster": <int>,  //可选，每个聚类的最少格式，可在初始化config中设置，默认值：2
        "endian": <string>,     //可选，特征二进制流的endian，取值只能为"big"或者"little",
                                //可在初始化config中设置，默认值："big"
        "incrementally": <int>    //可选，是否增量聚类，取值0或1，可在初始化config中设置，默认为1
    }
}
```
## 返回值格式
````json
{
    "code": 0,
    "message": "...",
    "result": <cluster result： json string>
}

<cluster result: json string>:
    [
        {
            "face_id": <string>,
            "group_id": <int>,
            "distance_to_center": <float>
        },
        {
            "face_id": <string>,
            "group_id": <int>
            "distance_to_center": <float>
        },
        ...
    ]
````

## 回归测试
1. [测试特征](http://oayjpradp.bkt.clouddn.com/face_cluster_feats_bin.zip)

2. 测试结果：
```json
 ([{'header': {'X-Origin-A': [':1']}, 'message': '', 'code': 0, 'result': '[{"distance_to_center": 0.173572, "face_id": "null", "group_id": 0}, {"distance_to_center": 0.209585, "face_id": "null", "group_id": 0}, {"distance_to_center": 0.2056, "face_id": "null", "group_id": 0}, {"distance_to_center": 0.189777, "face_id": "null", "group_id": 1}, {"distance_to_center": 0.165323, "face_id": "null", "group_id": 1}, {"distance_to_center": 0.231888, "face_id": "null", "group_id": 1}, {"distance_to_center": 0.159611, "face_id": "null", "group_id": 2}, {"distance_to_center": 0.156788, "face_id": "null", "group_id": 2}, {"distance_to_center": 0.200423, "face_id": "null", "group_id": 2}]'}], 0, '')
```


