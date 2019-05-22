
#include "infer.hpp"

#include <fstream>

#include <glog/logging.h>  // NOLINT
#include <opencv2/opencv.hpp>
#include "google/protobuf/text_format.h"

#include "common/archiver.hpp"
#include "common/errors.hpp"
#include "common/image.hpp"
#include "common/json.hpp"
#include "common/md5.hpp"
#include "common/protobuf.hpp"
#include "platform_shadow.hpp"
#include "proto/inference.pb.h"

struct CustomParams {
  CustomParams() {}
  ~CustomParams() {}

  int gpu_id;

  std::vector<std::string> labels_;
  int batch_size = MAX_BATCH_SIZE;

  bool const_use_quality = true;
  double neg_threshold = 0;
  double pose_threshold = 0;
  double cover_threshold = 0;
  double blur_threshold = 0.98;
  double quality_threshold = 0.6;
  bool output_quality_score = false;
  int min_face = 50;

  std::string models_prototxt = "/workspace/serving/models.prototxt";
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, CustomParams &p) {  // NOLINT
  ar.StartObject();

  ar.Member("gpu_id") & p.gpu_id;

  ar.Member("const_use_quality") & p.const_use_quality;
  ar.Member("blur_threshold") & p.blur_threshold;
  ar.Member("output_quality_score") & p.output_quality_score;
  ar.Member("min_face") & p.min_face;

  if (ar.HasMember("models_prototxt")) {
    ar.Member("models_prototxt") & p.models_prototxt;
  }
  return ar.EndObject();
}

inline std::string get_detect_json(
    bool output_quality, bool output_quality_score,
    const tron::fd::TronDetectionOutput &detection_output) {
  using namespace rapidjson;  // NOLINT
  Document document;
  auto &alloc = document.GetAllocator();
  Value j_detections(kObjectType), j_rects(kArrayType);
  for (const auto &rect : detection_output.objects) {
    Value j_rect(kObjectType);
    j_rect.AddMember("index", Value(rect.id), alloc);
    j_rect.AddMember("score", Value(rect.score), alloc);
    j_rect.AddMember("class", Value("face"), alloc);

    Value lt(kArrayType), rt(kArrayType), rb(kArrayType), lb(kArrayType),
        pts(kArrayType);
    lt.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymin), alloc);
    rt.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymin), alloc);
    rb.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymax), alloc);
    lb.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymax), alloc);
    pts.PushBack(lt, alloc).PushBack(rt, alloc).PushBack(rb, alloc).PushBack(
        lb, alloc);
    j_rect.AddMember("pts", pts, alloc);

    if (output_quality) {
      if (rect.quality_category == 0)
        j_rect.AddMember("quality", Value("clear"), alloc);
      if (rect.quality_category == 2)
        j_rect.AddMember("quality", Value("blur"), alloc);
      if (rect.quality_category == 3)
        j_rect.AddMember("quality", Value("pose"), alloc);
      if (rect.quality_category == 4)
        j_rect.AddMember("quality", Value("cover"), alloc);
      if (rect.quality_category == 5)
        j_rect.AddMember("quality", Value("small"), alloc);
      switch (rect.orient_category) {
        case 0:
          j_rect.AddMember("orientation", Value("up"), alloc);
          break;
        case 1:
          j_rect.AddMember("orientation", Value("up_left"), alloc);
          break;
        case 2:
          j_rect.AddMember("orientation", Value("left"), alloc);
          break;
        case 3:
          j_rect.AddMember("orientation", Value("down_left"), alloc);
          break;
        case 4:
          j_rect.AddMember("orientation", Value("down"), alloc);
          break;
        case 5:
          j_rect.AddMember("orientation", Value("down_right"), alloc);
          break;
        case 6:
          j_rect.AddMember("orientation", Value("right"), alloc);
          break;
        case 7:
          j_rect.AddMember("orientation", Value("up_right"), alloc);
          break;
      }
    }
    if (output_quality && output_quality_score && rect.quality_category != 1 &&
        rect.quality_category != 5) {
      Value q_score(kObjectType);
      q_score.AddMember("clear", Value(rect.quality_cls[0]), alloc);
      q_score.AddMember("blur", Value(rect.quality_cls[2]), alloc);
      q_score.AddMember("neg", Value(rect.quality_cls[1]), alloc);
      q_score.AddMember("cover", Value(rect.quality_cls[4]), alloc);
      q_score.AddMember("pose", Value(rect.quality_cls[3]), alloc);
      j_rect.AddMember("q_score", q_score, alloc);
    }
    j_rects.PushBack(j_rect, alloc);
  }
  j_detections.AddMember("detections", j_rects, alloc);
  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  j_detections.Accept(writer);
  return std::string(buffer.GetString());
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
  std::shared_ptr<tron::fd::Inference> inference_;
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

int QTPredCreate(const void *in_data, const int in_size,
                 PredictorContext *out) {
  google::InitGoogleLogging("QT");
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();

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
    tensord::core::RegisterPlatform(tron::fd::Shadow::Create, "shadow");
    tensord::core::RegisterPlatform(tron::fd::ShadowFD::Create, "shadow_fd");

    tensord::proto::ModelConfig _config;
    std::ifstream t(config.params.models_prototxt);
    std::string prototxt((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());
    auto ok = google::protobuf::TextFormat::ParseFromString(prototxt, &_config);
    CHECK(ok) << "parse conf";
    tensord::core::LoadModel(&_config);

    for (int i = 0; i < _config.instance_size(); i++) {
      _config.mutable_instance(i)->set_batchsize(config.batch_size);
    }

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

  auto inference_fd = std::make_shared<tron::fd::FDInference>();
  inference_fd->Setup(c->engines_->Get("fd", 0),
                      {c->config_.batch_size, 3, 512, 512});
  auto inference_qa = std::make_shared<tron::fd::QualityInference>();
  tron::fd::QualityInferenceConfig conf;
  conf.in_shape = {c->config_.batch_size, 3, 96, 96};
  conf.neg_threshold = c->config_.params.neg_threshold;
  conf.pose_threshold = c->config_.params.pose_threshold;
  conf.cover_threshold = c->config_.params.cover_threshold;
  conf.blur_threshold = c->config_.params.blur_threshold;
  conf.quality_threshold = c->config_.params.quality_threshold;
  inference_qa->Setup(c->engines_->Get("qa", 0),
                      {c->config_.batch_size, 3, 96, 96},
                      conf);
  auto inference = std::make_shared<tron::fd::Inference>();
  inference->fd_ = inference_fd;
  inference->qa_ = inference_qa;

  inference->labels_ = c->config_.params.labels_;
  inference->const_use_quality = c->config_.params.const_use_quality;
  inference->neg_threshold = c->config_.params.neg_threshold;
  inference->pose_threshold = c->config_.params.pose_threshold;
  inference->cover_threshold = c->config_.params.cover_threshold;
  inference->blur_threshold = c->config_.params.blur_threshold;
  inference->quality_threshold = c->config_.params.quality_threshold;
  inference->output_quality_score = c->config_.params.output_quality_score;
  inference->min_face = c->config_.params.min_face;

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

  // Parse InferenceRequests data
  inference::InferenceRequests requests;
  requests.ParseFromArray(in_data, in_size);

  inference::InferenceResponses responses_;
  for (int n = 0; n < requests.requests_size(); ++n) {
    const auto &req_data = requests.requests(n).data();
    auto *res = responses_.add_responses();
    if (req_data.has_body() && !req_data.body().empty()) {
      const auto &im_mat = tron::decode_image_buffer(req_data.body());
      if (!im_mat.empty()) {
        // Process image, preprocess, network forward, post postprocess
        bool use_quality = h->config_.params.const_use_quality;
        if (requests.requests(n).has_params()) {
          const std::string request_params = requests.requests(n).params();
          const auto document = tron::get_document(request_params);
          if (document.HasMember("use_quality")) {
            use_quality = document["use_quality"].GetInt();
          }
        }

        tron::fd::TronDetectionOutput detection_output = {};
        h->inference_->Predict(im_mat, &detection_output, use_quality);
        const auto &result_json_str = get_detect_json(
            use_quality,
            h->config_.params.output_quality_score,
            detection_output);
        // res->set_code(0);
        // res->set_message(tron::get_status_message(status));
        res->set_result(result_json_str);
        // _Error_ = const_cast<char *>(tron::get_status_message(status));
        // return status > 400 && status < 500 ? 400 : 500;
      } else {
        LOG(WARNING) << "OpenCV decode buffer error!";
        _Error_ = const_cast<char *>(
            tron::get_status_message(tron::tron_status_imdecode_error));
        return tron::tron_status_imdecode_error > 400 &&
                       tron::tron_status_imdecode_error < 500
                   ? 400
                   : 500;
      }
    } else {
      LOG(WARNING) << "RequestData body is empty!";
      _Error_ = const_cast<char *>(
          tron::get_status_message(tron::tron_status_request_data_body_empty));
      return tron::tron_status_request_data_body_empty > 400 &&
                     tron::tron_status_request_data_body_empty < 500
                 ? 400
                 : 500;
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
