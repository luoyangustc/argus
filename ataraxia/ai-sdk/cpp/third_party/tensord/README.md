# tensord

## lib形式

## 服务形式

## platform实现

编译方式
```
cmake -DPLATFORM_ALL=ON ..
or cmake -DPLATFORM_xx=ON -DPLATFORM_yy=ON ..
```

### 自定义platform

```cpp
#include "core/net.hpp"
#include "core/net_list.hpp"

class XX final : public tensord::core::Net<float> {
  void Init() override; 
  void Predict(const std::vector<NetIn<float>> &,
               std::vector<NetOut<float>> *) override;
  void Release() override;
}

tensord::core::RegisterPlatform(
    [](tensord::proto::Model model,
       tensord::proto::Instance::Kind,
       int,
       int) -> std::shared_ptr<tensord::core::Net<float>> { ... },
    "xx");

```

## 其他

### Docker

`resource`目录下有现成可用的`Dockerfile`，使用方式：
```
python tools/docker-build -f resource/xx.Dockerfile build_docker -t xx:yy
```
