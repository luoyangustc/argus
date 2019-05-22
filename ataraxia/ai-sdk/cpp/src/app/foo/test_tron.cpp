
#include "glog/logging.h"

#include "common/archiver.hpp"
#include "common/time.hpp"
#include "infer.hpp"
#include "proto/inference.pb.h"

int main(int argc, char const *argv[]) {
  std::vector<std::string> filenames;
  for (int i = 1; i < argc; i += 2) {
    if (std::strcmp(argv[i], "--image") == 0) filenames.push_back(argv[i + 1]);
  }

  int code;
  const char *err;

  PredictorContext ctx;
  PredictorHandle handle;

  {
    // Prepare protobuf's CreateParams
    inference::CreateParams create_params;
    const int batch_size = 8;
    create_params.set_batch_size(batch_size);

    const std::string custom_params = R"({
      "width":64, "height":64, "channel":3,
      "frontend":"tcp://127.0.0.1:5555",
      "backend":"tcp://127.0.0.1:5556"
      })";
    create_params.set_custom_params(custom_params);

    // Serialize protobuf's CreateParams to bytes
    std::vector<char> create_params_data(create_params.ByteSize());
    create_params.SerializeToArray(create_params_data.data(),
                                   create_params.ByteSize());

    code = QTPredCreate(create_params_data.data(), create_params.ByteSize(),
                        &ctx);
    if (code != 0) {
      err = QTGetLastError();
      LOG(ERROR) << code << "  " << std::string(err);
      return 1;
    }

    code = QTPredHandle(ctx,
                        create_params_data.data(), create_params.ByteSize(),
                        &handle);
    if (code != 0) {
      err = QTGetLastError();
      LOG(ERROR) << code << " " << std::string(err);
      return 1;
    }
  }

  inference::InferenceRequests inference_requests;
  for (auto cur = filenames.begin(); cur != filenames.end(); cur++) {
    // Read image from file to bytes
    auto *im_fp = fopen(cur->c_str(), "rb");
    fseek(im_fp, 0, SEEK_END);
    auto im_size = ftell(im_fp);
    rewind(im_fp);
    std::vector<char> image_data(im_size, 0);
    fread(image_data.data(), 1, im_size, im_fp);
    fclose(im_fp);

    auto request = inference_requests.add_requests();
    request->mutable_data()->set_body(image_data.data(), im_size);
  }

  auto inference_requests_size = inference_requests.ByteSize();
  std::vector<char> inference_requests_data(inference_requests_size, 0);
  inference_requests.SerializeToArray(inference_requests_data.data(),
                                      inference_requests_size);

  void *inference_responses_data;
  int inference_responses_size;
  code = QTPredInference(handle,
                         inference_requests_data.data(),
                         inference_requests_size,
                         &inference_responses_data,
                         &inference_responses_size);
  if (code != 0) {
    err = QTGetLastError();
    LOG(ERROR) << code << " " << std::string(err);
    return 1;
  }

  inference::InferenceResponses inference_responses;
  inference_responses.ParseFromArray(
      static_cast<char *>(inference_responses_data),
      inference_responses_size);
  for (int i = 0; i < inference_responses.responses_size(); i++) {
    auto resp = inference_responses.responses(i);
    LOG(INFO) << i << " " << resp.result();
  }

  QTPredFree(ctx);

  return 0;
}
