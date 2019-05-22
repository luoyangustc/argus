#pragma once

#include "tensord/proto/tensord.pb.h"

namespace tensord {
namespace core {

int LoadModel(proto::ModelConfig *);

}  // namespace core
}  // namespace tensord
