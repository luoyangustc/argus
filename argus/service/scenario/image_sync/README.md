
## 架构

### 业务框架

```ASCII
+-------------+  URI   +--------------+  IMG   +---------+     +------+
| HTTP Router | -----> | Image Parser | -----> | Service | --> | Eval |
+-------------+        +--------------+        +---------+     +------+
```

* **HTTP Router**: HTTP服务入口
* **Image Parser**: 包括拉取图片、图片预处理等
* **Service**: 业务逻辑，联合多个`Eval`提供完整的图片推理语义。
* **Eval**: 原子推理

### 支撑框架

* **配置**: 统一获取各中间件、各服务配置，结合默认值提供一致的配置项管理。
* **监控埋点**: `Image Parser`/`Service`/`Eval`之上分别设立埋点
* **API文档**: 在`HTTP Router`注册维护各`Service`文档，并统一输出

## 监控埋点

| Metric Name| Type | Tag |
| :--- | :--- | :--- |
| `qiniu_ai_image_argus_service_request_count` | `Counter` | `service`/`api` |
| `qiniu_ai_image_argus_service_response_time` | `Histogram` | `service`/`api`/`code` |
| `qiniu_ai_image_argus_sub_service_request_count` | `Counter` | `sub_service`/`service` |
| `qiniu_ai_image_argus_sub_service_response_time` | `Histogram` | `sub_service`/`service`/`code` |

## Eval模型实例管理

* `Service`维护准确的版本信息：包括Docker镜像、模型文件、配置参数等
* `Service`维护推荐的部署形式：GPU分布、实例数等

### 配置信息

[config.go](./config.go)

## API文档设计

本场景下，API文档为REST风格的API描述。

### 各API文档基本规格

> ### Path （ version ）
> > Message
>
> *Requset*
>
> ```
> Method path HTTP/1.1
> Content-Type: application/json
>
> {
> }
> ```
>
> *Response*
> 
> ```
> 200 ok
> Content-Type: application/json
>
> {
> }
> ```
>
> *请求参数说明*
>
> |字段|类型|描述|
> |:--- |:--- |:--- |
> 
> *返回参数说明*
>
> |字段|类型|描述|
> |:--- |:--- |:--- |
>
> *返回错误说明*
> |字段|类型|描述|
> |:--- |:--- |:--- |
