#include "common/infer.hpp"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/archiver.hpp"
#include "common/image.hpp"
#include "common/protobuf.hpp"
#include "helper.hpp"
#include "inference.hpp"
#include "platform_tensorrt.hpp"
#include "proto/inference.pb.h"
#include "tensord/tensord.hpp"

struct CustomParams {
  int gpu_id = 0;
  int model_num = 0;
  std::string models_prototxt = "/workspace/serving/models.prototxt";
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, CustomParams &p) {  // NOLINT
  ar.StartObject();
  ar.Member("gpu_id") & p.gpu_id;
  ar.Member("model_num") & p.model_num;
  if (ar.HasMember("models_prototxt")) {
    ar.Member("models_prototxt") & p.models_prototxt;
  }
  return ar.EndObject();
}

template <typename Archiver>
Archiver &operator&(Archiver &ar, tron::mix::ResponseMix &p) {  // NOLINT
  ar.StartObject();
  ar.Member("result");
  ar.StartObject();

  ar.Member("normal") & p.normal;
  ar.Member("march") & p.march;
  ar.Member("text") & p.text;
  ar.Member("face") & p.face;
  ar.Member("bk") & p.bk;
  ar.Member("pulp") & p.pulp;

  int ni = p.boxes.size();
  ar.Member("facenum") & ni;

  std::size_t n = p.boxes.size();
  std::size_t s4 = 4;
  std::size_t s2 = 2;
  std::size_t s128 = 128;

  ar.Member("faces");
  ar.StartArray(&n);
  for (std::size_t i = 0; i < p.boxes.size(); i++) {
    ar.StartObject();
    ar.Member("pts");
    int xmin = static_cast<int>(p.boxes[i].xmin),
        ymin = static_cast<int>(p.boxes[i].ymin),
        xmax = static_cast<int>(p.boxes[i].xmax),
        ymax = static_cast<int>(p.boxes[i].ymax);
    ar.StartArray(&s4);
    ar.StartArray(&s2) & xmin &ymin;
    ar.EndArray();
    ar.StartArray(&s2) & xmax &ymin;
    ar.EndArray();
    ar.StartArray(&s2) & xmax &ymax;
    ar.EndArray();
    ar.StartArray(&s2) & xmin &ymax;
    ar.EndArray();
    ar.EndArray();

    ar.Member("features");
    ar.StartArray(&s128);
    for (std::size_t j = 0; j < 128; j++) {
      ar &p.features[i][j];
    }
    ar.EndArray();
    ar.EndObject();
  }
  ar.EndArray();

  ar.EndObject();
  return ar.EndObject();
}

////////////////////////////////////////////////////////////////////////////////

class Config {
 public:
  int batch_size;
  CustomParams params;
};

class Handle {
 public:
  Handle() = default;

  Config config_;
  std::shared_ptr<tron::mix::Inference> inference_;
  std::vector<char> out_data_;
};

class Context {
 public:
  Context() = default;

  Config config_;
  std::shared_ptr<tensord::core::Engines> engines_;
  std::vector<Handle *> handles_;
};

////////////////////////////////////////////////////////////////////////////////

thread_local char *_Error_;
const char *QTGetLastError() { return _Error_; }

int QTPredCreate(const void *in_data,
                 const int in_size,
                 PredictorContext *out) {
  google::InitGoogleLogging("QT");
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();

  LOG(INFO) << "Starting createNet!";

  auto ctx = new Context();

  Config config;
  inference::CreateParams create_params;
  bool success = tron::read_proto_from_array(in_data, in_size, &create_params);
  if (!success) {
    LOG(ERROR) << "Parsing CreateParams Error! " << success;
    return 1;
  }
  config.batch_size = create_params.batch_size();
  {
    JsonReader jReader(create_params.mutable_custom_params()->c_str());
    jReader &config.params;
  }
  ctx->config_ = config;

  {
    tensord::core::RegisterPlatform(tron::mix::TensorRT::Create, "tensorrt");

    tensord::proto::ModelConfig _config;
    std::ifstream t(config.params.models_prototxt);
    std::string prototxt((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());
    auto ok = google::protobuf::TextFormat::ParseFromString(prototxt, &_config);
    CHECK_EQ(ok, true) << "Parse model config";
    tensord::core::LoadModel(&_config);

    // for (int i = 0; i < _config.instance_size(); i++) {
    //   _config.mutable_instance(i)->set_batchsize(config.batch_size);
    // }

    ctx->engines_ = std::make_shared<tensord::core::Engines>();
    ctx->engines_->Set(_config);
  }

  *out = ctx;
  LOG(INFO) << "Finished createNet!";
  return 0;
}

int QTPredHandle(PredictorContext ctx,
                 const void *, const int,
                 PredictorHandle *handle) {
  LOG(INFO) << "handle begin...";
  Context *c = reinterpret_cast<Context *>(ctx);
  auto h = new Handle();
  h->out_data_ = std::vector<char>(1024 * 1024 * 4);
  h->config_ = c->config_;

  auto inference_mix = std::make_shared<tron::mix::InferenceMix>();
  inference_mix->Setup(c->engines_->Get("mix", 0),
                       {c->config_.batch_size, 3, 224, 224});
  auto inference_fo = std::make_shared<tron::mix::InferenceFO>();
  inference_fo->Setup(c->engines_->Get("fo", 0),
                      {c->config_.batch_size, 3, 48, 48});
  auto inference_ff = std::make_shared<tron::mix::InferenceFF>();
  inference_ff->Setup(c->engines_->Get("ff", 0),
                      {c->config_.batch_size, 3, 112, 112});
  auto inference = std::make_shared<tron::mix::Inference>();
  inference->mix_ = inference_mix;
  inference->fo_ = inference_fo;
  inference->ff_ = inference_ff;

  h->inference_ = inference;
  c->handles_.push_back(h);
  *handle = h;
  LOG(INFO) << "handle done.";
  return 0;
}

int QTPredInference(PredictorHandle handle,
                    const void *in_data, const int in_size,
                    void **out_data, int *out_size) {
  LOG(INFO) << "netInference() start.";

  Handle *h = reinterpret_cast<Handle *>(handle);

  inference::InferenceRequests requests_;
  requests_.ParseFromArray(in_data, in_size);
  inference::InferenceResponses responses_;
  std::vector<cv::Mat> imgs_mat_;
  std::vector<std::string> imgs_attribute_;

  // normal processing
  for (int n = 0; n < requests_.requests_size(); ++n) {
    const auto &req_data = requests_.requests(n).data();
    auto *res = responses_.add_responses();
    res->set_code(0);
    if (req_data.has_body() && !req_data.body().empty()) {
      const auto &im_mat = tron::decode_image_buffer(req_data.body());
      const std::string im_attr = req_data.attribute();
      if (!im_mat.empty()) {
        if (im_mat.rows <= 1 || im_mat.cols <= 1) {  // image_size Error
          res->set_code(tron::mix::tron_error_round(
              tron::mix::tron_status_image_size_error));
          res->set_message(tron::mix::get_status_message(res->code()));
          continue;
        } else {
          imgs_mat_.push_back(im_mat);
          imgs_attribute_.push_back(im_attr);
          continue;
        }
      } else {  // decode Error
        res->set_code(
            tron::mix::tron_error_round(tron::mix::tron_status_imdecode_error));
        res->set_message(tron::mix::get_status_message(res->code()));
        continue;
      }
    } else {  // request_data_empty Error
      res->set_code(tron::mix::tron_error_round(
          tron::mix::tron_status_request_data_body_empty));
      res->set_message(tron::mix::get_status_message(res->code()));
      continue;
    }
  }

  std::vector<tron::mix::ResponseMix> results;
  h->inference_->Predict(imgs_mat_, &results);
  // set result_json_str
  int index = 0;
  for (int n = 0; n < requests_.requests_size(); ++n) {
    if (responses_.responses(n).code() == 0) {
      responses_.mutable_responses(n)->set_code(tron::mix::tron_status_success);
      responses_.mutable_responses(n)
          ->set_message(tron::mix::get_status_message(
              tron::mix::tron_status_success));
      JsonWriter writer;
      writer &results[index++];
      responses_.mutable_responses(n)->set_result(writer.GetString());
    }
  }

  auto size = responses_.ByteSize();
  CHECK_GE(h->out_data_.size(), size);
  responses_.SerializeToArray(&h->out_data_[0], responses_.ByteSize());
  *out_data = &h->out_data_[0];
  *out_size = responses_.ByteSize();
  LOG(INFO) << "inference() finished";
  return 0;
}

int QTPredFree(PredictorContext ctx) {
  Context *c = reinterpret_cast<Context *>(ctx);
  delete c;
  return 0;
}
