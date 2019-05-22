# Atlab Inference For Image Classify


## 关联issue
https://jira.qiniu.io/browse/ATLAB-5582

## 下载测试图片和范例模型

    ```
    sh data/get_data.sh
    sh models/get_models.sh
    ```

## 编译子工程Shadow及推理工程Tron
1. 运行一键编译脚本

    ```
    sh scripts/build_shell.sh
    ```

## 运行

    ```
    ./build/tron/test_tron
    ```
## 模型
48类分类模型，结果收缩到 5类;输出的5个类别是：
```
0 0_terror
1 1_pulp
2 2_march
3 3_text
4 4_normal
```


## 返回结果格式范例
分类结果

    ```json
    {
    "confidences": [
        {
            "index": 4, 
            "score": 0.9553675651550293, 
            "class": "normal"
        }
    ]
    }
    ```
