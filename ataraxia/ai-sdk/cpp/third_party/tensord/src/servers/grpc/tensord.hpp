#pragma once

#include <map>
#include <string>

#include "servers/grpc/tensord.grpc.pb.h"
#include "tensord/core/engines.hpp"

namespace tensord {
namespace server {

class GRPCImpl final : public grpc::Tensord::Service {
 public:
  GRPCImpl() = default;
  explicit GRPCImpl(std::shared_ptr<core::Engines> engines)
      : engines_(engines) {}

  ::grpc::Status Predict(::grpc::ServerContext*,
                         const grpc::Requests*,
                         grpc::Responses*) override;

 private:
  std::shared_ptr<core::Engines> engines_;
};

}  // namespace server
}  // namespace tensord
