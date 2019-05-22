
#include "tensord/core/load_model.hpp"

#include "tensord/core/utils.hpp"
#include "tensord/proto/tensord.pb.h"

namespace tensord {
namespace core {

int LoadModel(proto::ModelConfig *config) {
  auto dir = config->modelroot();

  for (int i = 0; i < config->model_size(); i++) {
    auto model = config->mutable_model(i);
    for (int j = 0; j < model->file_size(); j++) {
      auto file = model->mutable_file(j);
      if (file->mutable_body()->size() > 0) {
        continue;
      }
      auto name = file->name();
      if (file->mutable_alias()->size() > 0) {
        name = file->alias();
      }
      auto body = ReadFile(dir + "/" +
                           model->name() + "/" +
                           std::to_string(model->version()) + "/" +
                           name);
      file->mutable_body()->assign(
          reinterpret_cast<const char *>(&body[0]),
          body.size());
    }
  }

  return 0;
}

}  // namespace core
}  // namespace tensord
