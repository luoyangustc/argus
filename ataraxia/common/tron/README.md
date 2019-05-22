# Atlab Inference Example

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

## 返回结果格式范例
1. 通用检测

    ```json
    {
        "detections": [
            {
                "index": 1,
                "class": "guns",
                "score": 0.997133,
                "pts": [[225,195], [351,195], [351,389], [225,389]]
            },
            ...
        ]
    }
    ```

2. 通用分类

    ```json
    {
        "confidences": [
            {
                "index": 3,      
                "class": "Guns", 
                "score": 0.897
            },
            ...
        ]
    }
    ```

