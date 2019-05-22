
#include "servers/grpc/tensord.hpp"

#include <vector>

#include "glog/logging.h"

namespace tensord {
namespace server {

::grpc::Status GRPCImpl::Predict(::grpc::ServerContext *context,
                                 const grpc::Requests *requests,
                                 grpc::Responses *responses) {
  auto engine = engines_->Get(requests->model(), requests->version());
  if (engine.use_count() == 0) {
    return ::grpc::Status::CANCELLED;  // TODO(song)
  }

  std::vector<core::NetIn<float>> ins;
  for (int i = 0; i < requests->request_size(); i++) {
    auto request = requests->request(i);
    std::vector<std::string> names;
    std::vector<std::vector<float>> datas;
    for (int j = 0; j < request.data_size(); j++) {
      names.push_back(request.mutable_data(j)->name());
      auto body = request.mutable_data(j)->mutable_body();
      std::vector<float> data(body->size());
      memcpy(&data[0], body->data(), body->size());
      datas.push_back(data);
    }
    ins.emplace_back(names, datas);
  }
  std::vector<core::NetOut<float>> outs;

  engine->Predict(ins, &outs);

  for (std::size_t i = 0; i < outs.size(); i++) {
    auto response = responses->add_response();
    for (std::size_t j = 0; j < outs[i].datas.size(); j++) {
      auto data = response->add_data();
      data->set_name(outs[i].names[j]);
      data->mutable_body()->assign(
          reinterpret_cast<const char *>(&outs[i].datas[j][0]),
          outs[i].datas[j].size() * sizeof(float));
    }
  }

  return ::grpc::Status::OK;
}

}  // namespace server
}  // namespace tensord
