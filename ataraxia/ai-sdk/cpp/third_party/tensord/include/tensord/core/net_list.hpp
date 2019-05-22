#pragma once

#include <map>
#include <string>

#include "tensord/core/net.hpp"
#include "tensord/proto/tensord.pb.h"

namespace tensord {
namespace core {

typedef std::function<std::shared_ptr<Net<float>>(
    proto::Model,
    proto::Instance::Kind,
    int,
    int)>
    NewNetFunc;

std::map<std::string, NewNetFunc> AllPlatforms();
void RegisterPlatform(NewNetFunc, std::string);

}  // namespace core
}  // namespace tensord
