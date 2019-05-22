
#include <iostream>
#include <map>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "grpc++/grpc++.h"

#include "servers/grpc/tensord.hpp"
#include "tensord/core/engines.hpp"
#include "tensord/core/load_model.hpp"
#include "tensord/core/utils.hpp"

DEFINE_string(grpc_address, "127.0.0.1:50001", "grpc server address");
DEFINE_string(model_config, "model.prototxt", "config file in prototext");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<char> confBody;
  try {
    confBody = tensord::core::ReadFile(FLAGS_model_config);
  } catch (const std::invalid_argument& e) {
    LOG(ERROR) << "read model config file failed, path:" << FLAGS_model_config
               << ", error:" << e.what();
    std::exit(-1);
  }

  tensord::proto::ModelConfig config;
  auto ok = google::protobuf::TextFormat::ParseFromString(
      std::string(confBody.begin(), confBody.end()),
      &config);
  CHECK(ok) << "parse conf";
  tensord::core::LoadModel(&config);

  auto engines = std::make_shared<tensord::core::Engines>();
  engines->Set(config);

  // grpc server
  std::string server_address(FLAGS_grpc_address);
  tensord::server::GRPCImpl service(engines);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "GRPC Server listening on " << server_address;
  server->Wait();

  return 0;
}
