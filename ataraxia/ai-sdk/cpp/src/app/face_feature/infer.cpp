#include "infer.hpp"

#include <fstream>
#include <streambuf>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/text_format.h"

#include "common/archiver.hpp"
#include "common/errors.hpp"
#include "common/json.hpp"
#include "common/md5.hpp"
#include "common/protobuf.hpp"
#include "proto/inference.pb.h"

using namespace tron;  // NOLINT

struct CustomParams {
  CustomParams() {}
  ~CustomParams() {}

  int gpu_id = 0;
  int min_face_size = 50;
  int mirror_trick = 0;

  std::string models_prototxt = "/workspace/serving/models.prototxt";
  // V5: pre_fc1
  // V4: fc1
  std::string feature_output_layer = "pre_fc1";
  int batch_size_ff = 0;
  int batch_size_r = 0;
  int batch_size_o = 0;
  int batch_size_l = 0;
};

using tron::ff::FeatureConfig;
using tron::ff::Inference;

template <typename Archiver>
Archiver &operator&(Archiver &ar, CustomParams &p) {  // NOLINT
  ar.StartObject();
  if (ar.HasMember("gpu_id")) ar.Member("gpu_id") & p.gpu_id;

  if (ar.HasMember("mirror_trick")) ar.Member("mirror_trick") & p.mirror_trick;
  if (ar.HasMember("min_face_size")) {
    ar.Member("min_face_size") & p.min_face_size;
  }

  if (ar.HasMember("models_prototxt")) {
    ar.Member("models_prototxt") & p.models_prototxt;
  }
  if (ar.HasMember("feature_output_layer")) {
    ar.Member("feature_output_layer") & p.feature_output_layer;
  }
  if (ar.HasMember("batch_size_ff")) {
    ar.Member("batch_size_ff") & p.batch_size_ff;
  }
  if (ar.HasMember("batch_size_r")) ar.Member("batch_size_r") & p.batch_size_r;
  if (ar.HasMember("batch_size_o")) ar.Member("batch_size_o") & p.batch_size_o;
  if (ar.HasMember("batch_size_l")) ar.Member("batch_size_l") & p.batch_size_l;

  return ar.EndObject();
}

inline bool check_valid_box_pts(const int pts[4][2]) {
  if (pts[0][0] == pts[3][0] && pts[0][1] == pts[1][1] &&
      pts[1][0] == pts[2][0] && pts[2][1] == pts[3][1] &&
      pts[2][0] > pts[0][0] && pts[2][1] > pts[0][1]) {
    return true;
  }

  return false;
}

inline bool parse_request_boxes(const std::string &attribute,
                                tron::VecBoxF *boxes) {
  const auto &document = tron::get_document(attribute);
  LOG(INFO) << attribute;
  boxes->clear();
  // input_landmarks->clear();
  char pts_valid_msg[] =
      ("'attribute' in request data must be a valid json dict string,"
       " and has key 'pts'."
       " pts must be in the form as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."
       " all of x1, x2, y1, y2 can be parsed into int values."
       " And also must have (x2>x1 && y2>y1).");
  char landmarks_valid_msg[] =
      ("'attribute' in request data must be a valid json dict string,"
       " and has key 'landmarks'."
       " landmarks must be in the form as "
       "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]."
       " all of x1...x5, y1...y5 can be parsed float int values.");
  if (!document.HasMember("landmarks") && !document.HasMember("pts")) {
    LOG(ERROR) << "no pts or landmarks. " << pts_valid_msg
               << landmarks_valid_msg;
    return false;
  }
  if (document.HasMember("landmarks")) {
    const auto &landmarks = document["landmarks"];
    bool isArray = landmarks.IsArray();
    if (!isArray) {
      LOG(ERROR) << "landmarks must be Array. " << landmarks_valid_msg;
      return false;
    }
    float t_landmarks[10];
    try {
      const int size = landmarks.Size();
      if (size != 10) {
        return false;
      }
      for (int i = 0; i < 10; i++) {
        auto &temp = landmarks[i];
        t_landmarks[i] = temp.GetFloat();
      }  // for
    }    //  try
    catch (const std::exception &e) {
      LOG(ERROR) << "Exception when parsing input box landmarks. " << e.what()
                 << "\n"
                 << landmarks_valid_msg;
      return false;
    }  // catch
    // for (int i = 0; i < 10; i += 2) {
    //   input_landmarks->emplace_back(t_landmarks[i], t_landmarks[i + 1]);
    // }
    // has_landmarks_ = 1;
  }  // if
     // need  ignore in the post process
  if (document.HasMember("pts")) {
    const auto &pts = document["pts"];
    int t_pts[4][2];
    try {
      bool isArray = pts.IsArray();
      if (!isArray) {
        LOG(ERROR) << "pts must be Array. " << pts_valid_msg;
        return false;
      }
      const int size = pts.Size();
      if (size != 4) {
        return false;
      }
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
          t_pts[i][j] = pts[i][j].GetInt();
        }
      }
    } catch (...) {
      LOG(ERROR) << "Exception when parsing input box pts. " << pts_valid_msg;
      return false;
    }
    if (!check_valid_box_pts(t_pts)) {
      LOG(ERROR) << "check_valid_box_pts() failed. Invalid pts for a box. "
                 << pts_valid_msg;
      return false;
    }

    tron::BoxF box;
    box.xmin = t_pts[0][0];
    box.ymin = t_pts[0][1];
    box.xmax = t_pts[2][0];
    box.ymax = t_pts[2][1];
    boxes->push_back(box);
    return true;
  }

  return true;
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
  std::shared_ptr<Inference> inference_;
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

  LOG(INFO) << "facex-feature-tron: createNet() start";

  Config config;
  std::vector<std::vector<std::vector<char>>> net_param_datas;
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
    tensord::proto::ModelConfig _config;
    std::ifstream t(config.params.models_prototxt);
    std::string prototxt((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());
    auto ok = google::protobuf::TextFormat::ParseFromString(prototxt, &_config);
    CHECK(ok) << "parse conf";
    tensord::core::LoadModel(&_config);

    for (int i = 0; i < _config.instance_size(); i++) {
      if (_config.mutable_instance(i)->model() == "ff") {
        _config.mutable_instance(i)->set_batchsize(
            config.params.batch_size_ff > 0
                ? config.params.batch_size_ff
                : config.params.mirror_trick == 1
                      ? config.batch_size * 2
                      : config.batch_size);
      } else if (_config.mutable_instance(i)->model() == "r") {
        _config.mutable_instance(i)->set_batchsize(
            config.params.batch_size_r > 0
                ? config.params.batch_size_r
                : config.batch_size);
      } else if (_config.mutable_instance(i)->model() == "o") {
        _config.mutable_instance(i)->set_batchsize(
            config.params.batch_size_o > 0
                ? config.params.batch_size_o
                : config.batch_size);
      } else if (_config.mutable_instance(i)->model() == "l") {
        _config.mutable_instance(i)->set_batchsize(
            config.params.batch_size_l > 0
                ? config.params.batch_size_l
                : config.batch_size);
      }
    }

    ctx->engines_ = std::make_shared<tensord::core::Engines>();
    ctx->engines_->Set(_config);
  }

  *out = ctx;
  LOG(INFO) << "facex-feature-tron: createNet() finished";
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

  auto inference_ff = std::make_shared<tron::ff::FeatureInference>();
  inference_ff->Setup(c->engines_->Get("ff", 0),
                      {c->config_.params.batch_size_ff > 0
                           ? c->config_.params.batch_size_ff
                           : c->config_.params.mirror_trick == 1
                                 ? c->config_.batch_size * 2
                                 : c->config_.batch_size,
                       3, 112, 112},
                      FeatureConfig(h->config_.params.feature_output_layer));
  auto inference_o = std::make_shared<tron::ff::MTCNNOInference>();
  inference_o->Setup(c->engines_->Get("o", 0),
                     {c->config_.params.batch_size_o > 0
                          ? c->config_.params.batch_size_o
                          : c->config_.batch_size,
                      3, 48, 48});
  auto inference_l = std::make_shared<tron::ff::MTCNNLInference>();
  inference_l->Setup(c->engines_->Get("l", 0),
                     {c->config_.params.batch_size_l > 0
                          ? c->config_.params.batch_size_l
                          : c->config_.batch_size,
                      15, 24, 24});

  auto inference_mtcnn = std::make_shared<tron::ff::MTCNNInference>();
  // inference_mtcnn->SetupRNet(inference_r);
  inference_mtcnn->SetupONet(inference_o);
  inference_mtcnn->SetupLNet(inference_l);
  auto inference = std::make_shared<tron::ff::Inference>();
  inference->feature_ = inference_ff;
  inference->mtcnn_ = inference_mtcnn;

  h->inference_ = inference;
  c->handles_.push_back(h);
  *handle = h;
  LOG(INFO) << "handle done.";
  return 0;
}

int QTPredInference(PredictorHandle handle,
                    const void *in_data, const int in_size,
                    void **out_data, int *out_size) {
  LOG(INFO) << "facex-feature-tron: netInference() start";

  Handle *h_ = reinterpret_cast<Handle *>(handle);

  // Parse InferenceRequests data
  inference::InferenceRequests _requests;
  _requests.ParseFromArray(in_data, in_size);
  inference::InferenceResponses responses;

  for (int i = 0; i < _requests.requests_size(); i++) {
    auto request = _requests.requests(i);
    auto *resp = responses.add_responses();

    std::vector<cv::Point2d> cv_faces_points;

    const auto &req_data = request.data();
    if (!req_data.has_body() || req_data.body().empty()) {
      LOG(WARNING) << "RequestData body is empty!";
      auto code = tron_status_request_data_body_empty;
      _Error_ = const_cast<char *>(get_status_message(code));
      return code;
    }
    LOG(INFO) << "deconde image: " << req_data.body().size();
    auto im_mat = decode_image_buffer(req_data.body());
    if (im_mat.empty()) {
      LOG(WARNING) << "OpenCV decode buffer error!";
      auto code = tron_status_imdecode_error;
      _Error_ = const_cast<char *>(get_status_message(code));
      return code;
    }
    // Parse request face boxes from attribute
    VecBoxF face_boxes;
    // LOG(INFO)<<"req_data.attribute="<<req_data.attribute();
    bool success = parse_request_boxes(req_data.attribute(), &face_boxes);
    // LOG(INFO)<<"has_landmarks_="<<has_landmarks_;
    if (!success) {
      int code = 599;
      _Error_ = const_cast<char *>(get_status_message(code));
      return code;
    }

    const int image_w = im_mat.cols, image_h = im_mat.rows;
    BoxF box(0, 0, image_w, image_h);
    if (face_boxes.size() > 0) {
      box = face_boxes[0];
    }
    const int w = box.xmax - box.xmin;
    const int h = box.ymax - box.ymin;

    if (box.xmin < 0 || box.xmax > image_w || box.ymin < 0 ||
        box.ymax > image_h || w > image_w || h > image_h) {
      LOG(ERROR) << "face region must be in the image! ";
      auto code = tron_status_image_size_error;
      _Error_ = const_cast<char *>(get_status_message(code));
      return code;
    }

    // if (h < h_->params.min_face_size || w < h_->params.min_face_size) {
    //   LOG(ERROR) << "input rect width and height must be >="
    //              << h_->params.min_face_size;
    //   int code = 599;
    //   _Error_ = const_cast<char *>(get_status_message(code));
    //   return code;
    // }

    LOG(INFO) << "begin feature";
    std::vector<std::vector<float>> features;
    h_->inference_->Predict({im_mat},
                            {box},
                            {h_->config_.params.mirror_trick},
                            &features);
    const int sizeof_float = static_cast<int>(sizeof(float));
    LOG(INFO) << "end feature " << sizeof_float;
    for (size_t j = 0; j < features[0].size(); j++) {  // 临时兼容返回大端序
      float retVal;
      char *floatToConvert = reinterpret_cast<char *>(&features[0][j]);
      char *returnFloat = reinterpret_cast<char *>(&retVal);
      for (int k = 0; k < sizeof_float; k++) {
        returnFloat[k] = floatToConvert[sizeof_float - k - 1];
      }
      features[0][j] = retVal;
    }
    resp->mutable_body()->assign(std::string(
        reinterpret_cast<char *>(&features[0][0]),
        features[0].size() * sizeof_float));
  }  // for

  auto size = responses.ByteSize();
  CHECK_GE(h_->out_data_.size(), size);
  responses.SerializeToArray(&h_->out_data_[0], responses.ByteSize());
  *out_data = &h_->out_data_[0];
  *out_size = responses.ByteSize();

  LOG(INFO) << "facex-feature-tron: netInference() finished";
  return 0;
}

int QTPredFree(PredictorContext ctx) {
  Context *c = reinterpret_cast<Context *>(ctx);
  delete c;
  return 0;
}
