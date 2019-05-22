
#include <caffe/caffe.hpp>
#include "glog/logging.h"
#include "mxnet/c_predict_api.h"

#include "common/archiver.hpp"
#include "common/time.hpp"
#include "infer.hpp"
#include "proto/inference.pb.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

struct FF {
  FF() {}
  ~FF() {}

  double feature[512];
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, FF &ff) {
  ar.StartArray();
  for (size_t i = 0; i < 512; i++) {
    ar &ff.feature[i];
  }
  return ar.EndArray();
}

struct Pts {
  Pts() {}
  ~Pts() {}

  int pts[4][2];
};

template <typename Archiver>
Archiver &operator&(Archiver &ar, Pts &pts) {
  ar.StartArray();
  for (size_t i = 0; i < 4; i++) {
    ar.StartArray();
    ar &pts.pts[i][0];
    ar &pts.pts[i][1];
    ar.EndArray();
  }
  return ar.EndArray();
}

inline vector<char> readFile(std::string filename) {
  auto *fp = fopen(filename.c_str(), "rb");
  LOG(INFO) << "Read File: " << filename << " " << (fp == nullptr);
  fseek(fp, 0, SEEK_END);
  auto size = ftell(fp);
  rewind(fp);
  vector<char> data(size, 0);
  fread(data.data(), 1, size, fp);
  fclose(fp);
  return data;
}

int main(int argc, char const *argv[]) {
  // google::InitGoogleLogging(argv[0]);
  // google::SetStderrLogging(google::INFO);

  int code;
  const char *err;
  PredictorContext ctx;
  PredictorHandle handle;

  std::string model_dir;
  std::string image_dir;
  std::string image_tsv;

  cout << "-----------------------------------------------------" << endl;
  cout << "USAGE: " << endl;
  cout << argv[0] << "\n"
       << "--model_dir <model-path>\n"
       << "--image_dir <image-path>\n"
       << "--image_tsv <image-tsv-file>\n";
  cout << "-----------------------------------------------------" << endl;

  if (argc > 1) {
    for (int i = 1; i < argc; i += 2) {
      string t_str(argv[i]);
      if (t_str == "--model_dir") {
        model_dir = argv[i + 1];
      } else if (t_str == "--image_dir") {
        image_dir = argv[i + 1];
      } else if (t_str == "--image_tsv") {
        image_tsv = argv[i + 1];
      }
    }  // for
  }    // if

  cout << "-----------------------------------------------------" << endl;
  cout << "model_dir: " << model_dir << endl;
  cout << "image_dir: " << image_dir << endl;
  cout << "image_tsv: " << image_tsv << endl;
  cout << "-----------------------------------------------------" << endl;

  cout << "before create_params..." << endl;
  // Prepare protobuf's CreateParams
  inference::CreateParams create_params;
  const int batch_size = 4;
  create_params.set_batch_size(batch_size);

  std::vector<std::string> model_names = {
      model_dir + "feature-model/model-symbol.json",
      model_dir + "feature-model/model-0000.params",
      model_dir + "mtcnn-caffe-model/det2.prototxt",
      model_dir + "mtcnn-caffe-model/det2.caffemodel",
      model_dir + "mtcnn-caffe-model/det3.prototxt",
      model_dir + "mtcnn-caffe-model/det3.caffemodel",
      model_dir + "mtcnn-caffe-model/det4.prototxt",
      model_dir + "mtcnn-caffe-model/det4.caffemodel",
  };
  for (auto name : model_names) {
    std::vector<char> data = readFile(name);
    auto *_model = create_params.add_model_files();
    _model->set_name(name.c_str());
    _model->set_body(data.data(), data.size());
  }

  // add min_face_size, r-net,l-net configration
  const std::string custom_params =
      R"({
        "gpu_id": 0,
        "frontend_ff": "ipc://fronten_ff.ipc",
        "backend_ff": "ipc://backend_ff.ipc",
        "frontend_r": "ipc://fronten_r.ipc",
        "backend_r": "ipc://backend_r.ipc",
        "frontend_o": "ipc://fronten_o.ipc",
        "backend_o": "ipc://backend_o.ipc",
        "frontend_l": "ipc://fronten_l.ipc",
        "backend_l": "ipc://backend_l.ipc",
        "mirror_trick": 1,
        "min_face_size": 50
      })";
  create_params.set_custom_params(custom_params);

  // Serialize protobuf's CreateParams to bytes
  auto create_params_size = create_params.ByteSize();
  vector<char> create_params_data(create_params_size, 0);
  create_params.SerializeToArray(create_params_data.data(), create_params_size);

  cout << "\n---> createNet() starts..." << endl;
  // Do createNet
  code = QTPredCreate(create_params_data.data(), create_params.ByteSize(),
                      &ctx);
  if (code != 0) {
    err = QTGetLastError();
    LOG(ERROR) << code << " " << std::string(err);
    return 1;
  }
  code = QTPredHandle(ctx,
                      create_params_data.data(), create_params_size,
                      &handle);
  if (code != 0) {
    err = QTGetLastError();
    LOG(ERROR) << code << " " << std::string(err);
    return 1;
  }
  // cout<<"after createNet.\n";
  cout << "<--- createNet() finished..." << code << endl;

  std::ifstream in(image_tsv);

  std::vector<std::vector<float> > features;
  inference::InferenceRequests inference_requests;
  std::string line;
  while (getline(in, line)) {
    std::string::size_type i1(line.find_first_of('\t'));
    std::string name(line, 0, i1);
    std::string::size_type i2(line.find_first_of('\t', i1 + 1));
    std::string pts_str(line, i1 + 1, i2 - i1 - 1);
    std::string feature_str(line, i2 + 1);

    {
      FF ff;
      JsonReader reader(feature_str.c_str());
      reader &ff;
      std::vector<float> feature;
      for (int i = 0; i < 512; i++) {
        feature.push_back(ff.feature[i]);
      }
      features.push_back(feature);
    }

    LOG(INFO) << name << " -- " << name, pts_str;

    std::string im_file(image_dir + name);
    // Read image from file to bytes
    auto *im_fp = fopen(im_file.c_str(), "rb");
    CHECK_NOTNULL(im_fp);
    fseek(im_fp, 0, SEEK_END);
    auto im_size = ftell(im_fp);
    rewind(im_fp);
    vector<char> image_data(im_size, 0);
    fread(image_data.data(), 1, im_size, im_fp);
    fclose(im_fp);

    auto *request = inference_requests.add_requests();
    // request->add_data();
    request->mutable_data()->set_uri(im_file);
    request->mutable_data()->set_body(image_data.data(), im_size);
    request->mutable_data()->set_attribute("{\"pts\":" + pts_str + "}");
  }  // for

  // Serialize protobuf's InferenceRequests to bytes
  auto inference_requests_size = inference_requests.ByteSize();
  // std::cout<<"inference_requests_size="<<inference_requests_size<<"\n";
  vector<char> inference_requests_data(inference_requests_size, 0);
  inference_requests.SerializeToArray(inference_requests_data.data(),
                                      inference_requests_size);

  LOG(INFO) << "requests: " << inference_requests.requests_size() << " "
            << inference_requests.ByteSize();

  // Do netInference
  void *inference_responses_data;
  int inference_responses_size;

  Time t1;
  cout << "\n---> netInference() starts..." << endl;
  code = QTPredInference(handle,
                         inference_requests_data.data(),
                         inference_requests_size,
                         &inference_responses_data,
                         &inference_responses_size);
  cout << "<--- netInference() finished..." << endl;

  cout << "response code: " << code << endl;
  // cout << "response err info: " << err << endl;
  if (code != 200 && code != 0) {
    err = QTGetLastError();
    LOG(ERROR) << code << " " << std::string(err);
    return 1;
  }

  // const int FACE_FEATURE_SIZE=128;
  // Parse protobuf's InferenceResponses from bytes
  inference::InferenceResponses inference_responses;
  inference_responses.ParseFromArray(inference_responses_data,
                                     inference_responses_size);
  for (int i = 0; i < inference_responses.responses_size(); i++) {
    auto resp = inference_responses.responses(i);
    vector<float> ff(resp.mutable_body()->size() / sizeof(float));
    memcpy(&(ff[0]), resp.mutable_body()->data(),
           resp.mutable_body()->size());
    for (size_t j = 0; j < ff.size(); j++) {  // 临时兼容返回大端序
      float retVal;
      char *floatToConvert = reinterpret_cast<char *>(&features[0][j]);
      char *returnFloat = reinterpret_cast<char *>(&retVal);
      for (std::size_t k = 0; k < static_cast<int>(sizeof(float)); k++) {
        returnFloat[k] = floatToConvert[sizeof(float) - k - 1];
      }
      ff[j] = retVal;
    }

    float sum = 0;
    for (std::size_t j = 0; j < ff.size(); j++) {
      sum += ff[j] * features[i][j];
    }
    LOG(INFO) << sum;
  }
  // std::cout << inference_responses.responses(0).result() << std::endl;

  QTPredFree(ctx);

  return 0;
}
