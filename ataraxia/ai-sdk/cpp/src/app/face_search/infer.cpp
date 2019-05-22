#include "infer.hpp"

#include <algorithm>
#include <string>

#include "glog/logging.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/text_format.h"

#include "common/archiver.hpp"
#include "common/errors.hpp"
#include "common/md5.hpp"
#include "common/type.hpp"
#include "proto/inference.pb.h"

template <typename Archiver>
Archiver &operator&(Archiver &ar, tron::fs::SetParams &p) {  // NOLINT
  ar.StartObject();
  ar.Member("key") & p.key;
  ar.Member("size_limit") & p.size_limit;
  ar.Member("threshold");
  ar.StartArray();
  ar &p.threshold[0];
  ar &p.threshold[1];
  ar &p.threshold[2];
  ar.EndArray();
  return ar.EndObject();
}

template <typename Archiver>
Archiver &operator&(Archiver &ar, tron::fs::CustomParams &p) {  // NOLINT
  ar.StartObject();
  if (ar.HasMember("sets")) {
    ar.Member("sets");
    size_t size = 0;
    ar.StartArray(&size);
    for (std::size_t i = 0; i < size; i++) {
      tron::fs::SetParams set_p;
      ar &set_p;
      p.sets[set_p.key] = set_p;
    }
    ar.EndArray();
  }
  if (ar.HasMember("threshold")) {
    ar.Member("threshold");
    ar.StartArray();
    ar &p.threshold[0];
    ar &p.threshold[1];
    ar &p.threshold[2];
    ar.EndArray();
  }
  return ar.EndObject();
}

inline bool read_proto_from_array(const void *proto_data, int proto_size,
                                  google::protobuf::Message *proto) {
  return proto->ParseFromArray(proto_data, proto_size);
  //   using google::protobuf::io::ArrayInputStream;
  //   using google::protobuf::io::CodedInputStream;
  //   auto *param_array_input = new ArrayInputStream(proto_data, proto_size);
  //   auto *param_coded_input = new CodedInputStream(param_array_input);
  //   param_coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
  //   bool success = proto->ParseFromCodedStream(param_coded_input) &&
  //                  param_coded_input->ConsumedEntireMessage();
  //   delete param_coded_input;
  //   delete param_array_input;
  //   return success;
}

inline bool check_valid_box_pts(const int pts[4][2]) {
  if (pts[0][0] == pts[3][0] && pts[0][1] == pts[1][1] &&
      pts[1][0] == pts[2][0] && pts[2][1] == pts[3][1] &&
      pts[2][0] > pts[0][0] && pts[2][1] > pts[0][1]) {
    return true;
  }

  return false;
}

template <typename Archiver>
Archiver &operator&(Archiver &ar, std::array<std::array<int, 2>, 4> &pts) {  // NOLINT
  ar.StartObject();
  if (ar.HasMember("pts")) {
    ar.Member("pts");
    ar.StartArray();
    for (int i = 0; i < 4; i++) {
      ar.StartArray();
      for (int j = 0; j < 2; j++) {
        ar &pts[i][j];
      }
      ar.EndArray();
    }
    ar.EndArray();
  }
  return ar.EndObject();
}

template <typename Archiver>
Archiver &operator&(Archiver &ar, tron::fs::Feature &feature) {  // NOLINT
  ar.StartObject();
  if (ar.HasMember("size")) {
    ar.Member("size") & feature.key;
  }
  ar.Member("index") & feature.index;
  if (ar.HasMember("group")) {
    ar.Member("group") & feature.group;
  }
  ar.Member("feature");
  ar.StartArray();
  for (int i = 0; i < 512; i++) {
    ar &feature.feature[i];
  }
  ar.EndArray();

  if (ar.HasMember("url")) ar.Member("url") & feature.sample_url;
  if (ar.HasMember("id")) ar.Member("id") & feature.sample_id;
  if (ar.HasMember("pts")) {
    ar.Member("pts");
    ar.StartArray();
    for (int i = 0; i < 4; i++) {
      ar.StartArray();
      for (int j = 0; j < 2; j++) {
        ar &feature.sample_pts[i][j];
      }
      ar.EndArray();
    }
    ar.EndArray();
  }

  return ar.EndObject();
}

struct ConfidenceRet {
  ConfidenceRet() {}
  ConfidenceRet(int index, std::string clas, std::string group, float score)
      : index(index), clas(clas), group(group), score(score) {}
  ConfidenceRet(int index, std::string clas, std::string group, float score,
                std::string &sample_url, std::string &sample_id,
                std::array<std::array<int, 2>, 4> &sample_pts)
      : index(index),
        clas(clas),
        group(group),
        score(score),
        sample_url(sample_url),
        sample_pts(sample_pts),
        sample_id(sample_id) {}
  ~ConfidenceRet() {}

  int index;
  std::string clas;
  std::string group;
  double score;

  std::string sample_url;
  std::array<std::array<int, 2>, 4> sample_pts;
  std::string sample_id;
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, ConfidenceRet &p) {  // NOLINT
  ar.StartObject();
  ar.Member("index") & p.index;
  ar.Member("class") & p.clas;
  ar.Member("group") & p.group;
  ar.Member("score") & p.score;

  ar.Member("sample");
  ar.StartObject();
  ar.Member("url") & p.sample_url;
  ar.Member("id") & p.sample_id;
  ar.Member("pts");
  ar.StartArray();
  for (int i = 0; i < 4; i++) {
    ar.StartArray();
    for (int j = 0; j < 2; j++) {
      ar &p.sample_pts[i][j];
    }
    ar.EndArray();
  }
  ar.EndArray();
  ar.EndObject();

  return ar.EndObject();
}

struct Ret {
  Ret() {}
  ~Ret() {}

  std::vector<ConfidenceRet> confidences;
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, Ret &p) {  // NOLINT
  ar.StartObject();
  ar.Member("confidences");
  ar.StartArray();
  for (std::size_t i = 0; i < p.confidences.size(); i++) {
    ar &p.confidences[i];
  }
  ar.EndArray();

  return ar.EndObject();
}

////////////////////////////////////////////////////////////////////////////////

thread_local char *_Error_;
const char *QTGetLastError() { return _Error_; }

int QTPredCreate(const void *in_data, const int in_size,
                 PredictorContext *out) {
  google::InitGoogleLogging("QT");
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();

  LOG(INFO) << "facex-search: createNet() start";

  // Parse CreateParams data

  inference::CreateParams create_params;
  bool success = read_proto_from_array(in_data, in_size, &create_params);
  if (!success) {
    LOG(FATAL) << "Parsing CreateParams Error!";
    return 599;
  }

  auto *ctx = new tron::fs::Context();
  {
    ctx->params.sets[""] = tron::fs::SetParams();
  }
  std::string custom_params_str = create_params.custom_params();
  LOG(INFO) << custom_params_str;
  {
    ctx->params.threshold = {0.35, 0.375, 0.4};
    ctx->params.sets["XSmall"] =
        tron::fs::SetParams("XSmall", 24, {0.38, 0.4, 0.42});
    ctx->params.sets["small"] =
        tron::fs::SetParams("small", 32, {0.38, 0.4, 0.42});
    ctx->params.sets["large"] =
        tron::fs::SetParams("large", 60, {0.35, 0.375, 0.4});
  }
  {
    JsonReader jReader(custom_params_str.c_str());
    jReader & ctx->params;
  }

  ctx->batch_size = create_params.batch_size();

  int features_index, labels_index;
  for (int i = 0; i < create_params.custom_files_size(); i++) {
    auto file = create_params.custom_files(i);
    if (!file.has_name() || !file.has_body()) continue;
    auto name = file.name();
    auto body = file.mutable_body();
    LOG(INFO) << name;
    if (name.find("features.line") != std::string::npos) {
      LOG(INFO) << "file's body size: " << body->size();
      LOG(INFO) << "file's body md5: " << tron::MD5(*body);
      features_index = i;
    } else if (name.find("labels.line") != std::string::npos) {
      LOG(INFO) << "Model file's body size: " << body->size();
      LOG(INFO) << "Model file's body md5: " << tron::MD5(*body);
      labels_index = i;
    }
  }
  auto features_file = create_params.custom_files(features_index);
  auto features_body = features_file.mutable_body();
  auto labels_file = create_params.custom_files(labels_index);
  auto labels_body = labels_file.mutable_body();

  std::vector<std::string> labels;
  {
    std::string::size_type pos1 = 0;
    std::string::size_type pos2 = labels_body->find('\n');
    while (std::string::npos != pos2) {
      labels.push_back(labels_body->substr(pos1, pos2 - pos1));
      pos1 = pos2 + 1;
      pos2 = labels_body->find('\n', pos1);
    }
    labels.push_back(labels_body->substr(pos1));
  }

  {
    std::string::size_type pos1 = 0, pos2;
    do {
      pos2 = features_body->find('\n', pos1);

      tron::fs::Feature feature;
      if (std::string::npos != pos2) {
        JsonReader jReader(features_body->substr(pos1, pos2 - pos1).c_str());
        jReader &feature;
      } else {
        JsonReader jReader(features_body->substr(pos1).c_str());
        jReader &feature;
      }
      feature.label = labels[feature.index];
      int limit = ctx->params.sets[feature.key].size_limit;
      bool found = false;
      for (auto cur = ctx->features.begin();
           cur != ctx->features.end();
           cur++) {
        if (cur->first != limit) continue;
        cur->second.push_back(feature);
        found = true;
        break;
      }
      if (!found) {
        ctx->features
            .emplace_back(limit, std::vector<tron::fs::Feature>{feature});
      }

      if (std::string::npos != pos2) pos1 = pos2 + 1;
    } while (std::string::npos != pos2);
  }

  std::sort(ctx->features.begin(), ctx->features.end(),
            [](const std::pair<int, std::vector<tron::fs::Feature>> &pair1,
               const std::pair<int, std::vector<tron::fs::Feature>> &pair2)
                -> bool {
              return pair1.first < pair2.first;
            });
  for (auto cur = ctx->features.begin();
       cur != ctx->features.end(); cur++) {
    LOG(INFO) << cur->first << " " << cur->second.size();
  }

  *out = ctx;
  LOG(INFO) << "facex-search: createNet() finished";
  return 0;
}

int QTPredHandle(PredictorContext ctx,
                 const void *in_data, const int in_size,
                 PredictorHandle *handle) {
  LOG(INFO) << "handle begin...";

  tron::fs::Context *c = reinterpret_cast<tron::fs::Context *>(ctx);
  auto h = new tron::fs::Handle();
  h->out_data = std::vector<char>(1024 * 1024 * 4);
  h->batch_size = c->batch_size;
  h->params = &c->params;
  h->features = &c->features;
  c->handles.push_back(h);
  *handle = h;
  LOG(INFO) << "handle done.";
  return 0;
}

int QTPredInference(PredictorHandle handle,
                    const void *in_data, const int in_size,
                    void **out_data, int *out_size) {
  LOG(INFO) << "facex-feature-tron: netInference() start";

  tron::fs::Handle *h_ = reinterpret_cast<tron::fs::Handle *>(handle);

  // Parse InferenceRequests data
  inference::InferenceRequests _requests;
  _requests.ParseFromArray(in_data, in_size);
  inference::InferenceResponses responses;

  for (int i = 0; i < _requests.requests_size(); i++) {
    auto request = _requests.requests(i);
    auto *resp = responses.add_responses();

    if (request.mutable_data()->mutable_body()->size() != 512 * 4) {
      _Error_ = const_cast<char *>("failed to get feature data");
      return 400;
    }

    std::array<std::array<int, 2>, 4> pts;
    {
      JsonReader jReader(request.mutable_data()->mutable_attribute()->c_str());
      jReader &pts;
    }
    int side = (pts[2][0] - pts[0][0]) < (pts[2][1] - pts[0][1])
                   ? pts[2][0] - pts[0][0]
                   : pts[2][1] - pts[0][1];
    auto features = std::pair<int, std::vector<tron::fs::Feature>>(
        0, std::vector<tron::fs::Feature>());
    for (std::size_t j = 0; j < h_->features->size(); j++) {
      if (h_->features->at(j).first > side) {
        break;
      }
      features = h_->features->at(j);
    }

    LOG(INFO) << side << " "
              << features.first << " " << features.second.size();

    Ret ret;
    if (features.first > 0 &&
        request.mutable_data()->mutable_body()->size() == 512 * 4) {
      std::vector<float> req_feature(512);
      memcpy(&req_feature[0],
             request.mutable_data()->mutable_body()->data(),
             request.mutable_data()->mutable_body()->size());
      for (size_t j = 0; j < 512; j++) {
        float retVal;
        char *floatToConvert = reinterpret_cast<char *>(&req_feature[j]);
        char *returnFloat = reinterpret_cast<char *>(&retVal);
        returnFloat[0] = floatToConvert[3];
        returnFloat[1] = floatToConvert[2];
        returnFloat[2] = floatToConvert[1];
        returnFloat[3] = floatToConvert[0];
        req_feature[j] = retVal;
      }
      std::vector<ConfidenceRet> rets;
      for (auto cur = features.second.begin();
           cur != features.second.end(); cur++) {
        float score = 0;
        for (size_t j = 0; j < 512; j++) {
          score += cur->feature[j] * req_feature[j];
        }
        std::array<double, 3> &thresholds =
            h_->params->sets[cur->key].threshold;
        if (score < thresholds[1]) continue;
        if (score < thresholds[2]) {
          score = h_->params->threshold[1] +
                  (h_->params->threshold[2] - h_->params->threshold[1]) *
                      (score - thresholds[1]) /
                      (thresholds[2] - thresholds[1]);
        } else {
          score = h_->params->threshold[2] +
                  (1 - h_->params->threshold[2]) *
                      (score - thresholds[2]) /
                      (1 - thresholds[2]);
        }
        rets.emplace_back(cur->index, cur->label, cur->group, score,
                          cur->sample_url, cur->sample_id, cur->sample_pts);
      }
      std::sort(rets.begin(), rets.end(),
                [](const ConfidenceRet &ret1,
                   const ConfidenceRet &ret2) -> bool {
                  return ret1.score > ret2.score;
                });
      if (rets.size() > 0) {
        ret.confidences =
            std::vector<ConfidenceRet>(rets.begin(), rets.begin() + 1);
      }
    }

    JsonWriter writer;
    writer &ret;
    resp->set_result(writer.GetString());
  }  // for

  // Check responses buffer size must be not larger than 4M bytes
  int responses_size = responses.ByteSize();
  if (responses_size <= 1024 * 1024 * 4) {
    responses.SerializeToArray(&h_->out_data[0], responses_size);
    *out_size = responses_size;
    *out_data = &h_->out_data[0];
  } else {
    LOG(WARNING) << "Responses buffer size request for "
                 << responses_size / (1024 * 1024) << "MB";
    auto code = 599;
    _Error_ = const_cast<char *>("");
    return code;
  }

  LOG(INFO) << "facex-search: netInference() finished";
  return 0;
}

int QTPredFree(PredictorContext ctx) { return 0; }
