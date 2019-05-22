
算法Serve的依赖及工具

# Serving集成

推理模型按照既定的接口，以动态库（.so）的形式提供给工程服务集成。

## 集成接口

* `initEnv`：系统初始化，程序启动时执行一次。
* `createNet`：模型初始化初始化后模型常驻内存至程序退出。可能调用多次，表示初始化多个模型实例。
* `netPreprocess`：数据预处理，每次推理会先尝试执行预处理。可选实现。
* `netInference`：数据推理。

### 接口数据交换

所有输入输出的复杂数据均通过`protobuf2`进行序列化

### Python集成接口

参见[3.7.python-inference-image.md](../../doc/atlab-tutorial/3.developtutor/3.7.python-inference-image.md)

## 集成发布流程

### 1. 动态库（.so)开发

### 2. Docker Image制作
包含但不限于：模型运行环境、动态库（.so）

* 动态库路径: `/workspace/serving/inference.so`

### 3. push Docker Image至指定地址

* hub: reg.qiniu.com
* user: ava-public@qiniu.com
* namespace: inference

### 4. issue提交配置

```
{
    "model_file": "http://xxx/xxx.tar",
    "custom_files": {
        "xxx": "http://xxx/xxx",
        "yyy": "http://xxx/xxx"
    },
    "custom_params": {
        "xxx": xxx,
        "yyy": yyy
    }
}
```

### 建议

* 镜像制作在同一环境（Jenkins），基于入库（`github`）的`Dockerfile`，集成库中的代码
