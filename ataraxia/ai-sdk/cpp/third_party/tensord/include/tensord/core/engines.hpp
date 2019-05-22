#pragma once

#include <map>
#include <string>

#include "tensord/core/engine.hpp"

namespace tensord {
namespace core {

// TODO(song) 当前只实现float的支持
class Engines {
  using EngineM = std::map<std::string, std::shared_ptr<Engine<float>>>;

 public:
  void Set(const proto::ModelConfig &);
  void Set(std::string name,
           int version,
           std::shared_ptr<Engine<float>> engine);
  std::shared_ptr<Engine<float>> Get(std::string name, int version);

 private:
  static std::string EngineKey(std::string name, int version);

  EngineM engines_;
};

}  // namespace core
}  // namespace tensord
