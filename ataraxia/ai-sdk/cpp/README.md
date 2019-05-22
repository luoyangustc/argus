
# 引擎框架

框架包括三个模块

* 业务逻辑（inference）
* 调度器（mq)
* 模型推理（forward）

## 业务逻辑（inference）

* **输入**：单张图片
* **输出**：图片业务推算结果

核心细节
* 支持并行
* 单图流程为串行处理

## 调度器（mq)

* **输入**：单网络模型输入
* **输出**：单网络模型输出

核心细节
* 支持异步
* 支持batch
* 多实例调度

## 模型推理（forward） 

* **输入**：batch网络模型输入
* **输出**：batch网络模型输出

核心细节
* 串行循环

### 接口形式

```C++
void Forward(const std::vector<std::pair<size_t, const void *>> &,
             std::vector<std::vector<unsigned char>> &);
```

*人脸检测*
```C++
void Forward(const std::vector<inference::fd::ForwardRequest> &,
             std::vector<inference::fd::ForwardResponse> &);
```

## 流程示例（敏感人物识别）

```ascii
+------+     +-------------+     +------------------+     +------------------+     +---------------------+     +-------------+
| IMG1 | --> | decode img1 | --> | face detect IMG1 | --> | N * feature IMG1 | --> | N * politician IMG1 | --> | result IMG1 |
+------+     +-------------+     +------------------+     +------------------+     +---------------------+     +-------------+
                                   |                        |                        |
                                   | queue                  | queue                  | queue
                                   v                        v                        v
+------+     +-------------+     +------------------+     +------------------+     +---------------------+
| IMG2 | --> | decode img2 |     | 2 * face detect  |     |  N+M * feature   |     |  N+M * politician   |
+------+     +-------------+     +------------------+     +------------------+     +---------------------+
               |                   ^                        ^                        ^
               |                   | queue                  | queue                  | queue
               |                   |                        |                        |
               |                 +------------------+     +------------------+     +---------------------+     +-------------+
               +---------------> | face detect IMG2 | --> | M * feature IMG2 | --> | M * politician IMG2 | --> | result IMG2 |
                                 +------------------+     +------------------+     +---------------------+     +-------------+
```