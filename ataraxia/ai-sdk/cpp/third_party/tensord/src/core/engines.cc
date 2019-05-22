
#include "tensord/core/engines.hpp"

#include "core/engine_impl.hpp"
#include "tensord/core/net_list.hpp"

namespace tensord {
namespace core {

void Engines::Set(const proto::ModelConfig &config) {
  auto platforms = AllPlatforms();
  for (int i = 0; i < config.instance_size(); i++) {
    auto instance = config.instance(i);
    int j = 0;
    for (; j < config.model_size(); j++) {
      if (config.model(j).name() == instance.model() &&
          config.model(j).version() == instance.version()) {
        break;
      }
    }
    if (j >= config.model_size()) {
      return;  // TODO(song): return error
    }
    auto model = config.model(j);
    auto iter = platforms.find(model.platform());
    if (iter == platforms.end()) {
      return;  // TODO(song): return error
    }
    auto func = iter->second;

    auto engine = std::make_shared<EngineImpl<float>>();
    engine->Setup(func, model, instance);
    Set(instance.model(), instance.version(), engine);
  }
  return;
}

void Engines::Set(std::string name,
                  int version,
                  std::shared_ptr<Engine<float>> engine) {
  engines_[EngineKey(name, version)] = engine;
}

std::shared_ptr<Engine<float>> Engines::Get(std::string name, int version) {
  auto iter = engines_.find(EngineKey(name, version));
  if (iter == engines_.end()) {
    return std::make_shared<EngineImpl<float>>();
  }
  return iter->second;
}

std::string Engines::EngineKey(std::string name, int version) {
  return name + "-" + std::to_string(version);
}

}  // namespace core
}  // namespace tensord
