#include "process.hpp"
#include <boost/format.hpp>
#include <initializer_list>
#include <opencv2/opencv.hpp>
#include "common/errors.hpp"
#include "common/image.hpp"
#include "common/md5.hpp"
#include "debug_print.hpp"
#include "framework/context.hpp"
#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/message_differencer.h"
#include "gsl/gsl"
#include "gtest/gtest.h"
#include "infer.hpp"
#include "post_process.hpp"
#include "pre_process.hpp"
#include "proto/inference.pb.h"
#include "utils.hpp"

namespace tron {
namespace terror_mixup {
using std::vector;
using std::chrono::high_resolution_clock;

const string model_dir =  // NOLINT
    "../../res/model/ava-terror-mixup_terror-mixup-1121-11/";

std::string getEnvVar(std::string const &key) {
  char *val = getenv(key.c_str());
  return val == NULL ? std::string("") : std::string(val);
}

string format_ns(std::chrono::nanoseconds t) {
  return std::to_string(t.count() / 1000000.) + "ms";
}

TEST(TerrorMixup, TsvTest) {
  const int batch_size = 4;
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  {
    const string s = getEnvVar("GPU_INDEX");
    if (!s.empty()) {
      LOG(INFO) << "use gpu:" << std::stoi(s);
      caffe::Caffe::SetDevice(std::stoi(s));
    }
  }

  TerrorMixupDet terrorMixupDet(
      read_bin_file_to_string(model_dir + "det_deploy.prototxt"),
      read_bin_file_to_string(model_dir + "det_weight.caffemodel"), batch_size);

  TerrorMixupFine terrorMixupFine(
      read_bin_file_to_string(model_dir + "fine_deploy.prototxt"),
      read_bin_file_to_string(model_dir + "fine_weight.caffemodel"),
      batch_size);
  TerrorMixupCoarse terrorMixupCoarse(
      read_bin_file_to_string(model_dir + "coarse_deploy.prototxt"),
      read_bin_file_to_string(model_dir + "coarse_weight.caffemodel"),
      batch_size);

  post_process_param param;
  {
    param.fine_labels = read_bin_file_to_string(model_dir + "fine_labels.csv");
    param.det_labels = read_bin_file_to_string(model_dir + "det_labels.csv");
    param.coarse_labels =
        read_bin_file_to_string(model_dir + "coarse_labels.csv");
    param.percent_fine = 0.6;
    param.percent_coarse = 0.4;
    param.batch_size = batch_size;
  }
  const PostProcess p(param);

  const auto tsv =
      csv_parse(read_bin_file_to_string(
                    "../../res/testdata/image/serving/terror-mixup/"
                    "terror-mixup-201811211548/set20181108/201811211548.tsv"),
                "\t");
  for (std::size_t tsv_index = 0; tsv_index < tsv.size();
       tsv_index += batch_size) {
    _DEBUG_PRINT(tsv_index);
    vector<vector<string>> tsv_lines;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      std::size_t index = tsv_index + batch_index;
      if (index >= tsv.size()) {
        index = tsv.size() - 1;
      }
      tsv_lines.push_back(tsv[index]);
    }

    vector<vector<float>> det_input_s;
    vector<vector<float>> cls_input_s;
    vector<int> image_widths;
    vector<int> image_heights;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      const auto file_name = tsv_lines.at(batch_index)[0];
      const auto expect_response_text = tsv_lines.at(batch_index)[1];
      const auto file_path =
          "../../res/testdata/image/serving/terror-mixup/set20181108/" +
          file_name;
      cv::Mat im_ori = cv::imread(file_path, cv::IMREAD_COLOR);
      CHECK(im_ori.data);
      const auto image_width = im_ori.size().width;
      const auto image_height = im_ori.size().height;
      image_widths.push_back(image_width);
      image_heights.push_back(image_height);
      const vector<float> det_input = det_pre_process_image(im_ori);
      const vector<float> cls_input = cls_pre_process_image(im_ori);
      det_input_s.push_back(det_input);
      cls_input_s.push_back(cls_input);
    }

    const auto det_input_batch = join_batch_size_data(det_input_s);
    const auto cls_input_batch = join_batch_size_data(cls_input_s);
    input_det_shape.assert_shape_match(det_input_batch, batch_size);
    input_fine_shape.assert_shape_match(cls_input_batch, batch_size);
    input_coarse_shape.assert_shape_match(cls_input_batch, batch_size);

    auto start = high_resolution_clock::now();
    const auto output_det = terrorMixupDet.forward(det_input_batch);
    const auto output_fine = terrorMixupFine.forward(cls_input_batch);
    const auto output_coarse = terrorMixupCoarse.forward(cls_input_batch);
    auto end = high_resolution_clock::now();
    LOG(INFO) << "forward batch_size:" << batch_size
              << " use time:" << format_ns(end - start)
              << " qps:" << 1e9 / (end - start).count() * batch_size;

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      const auto expect_response_text = tsv_lines[batch_index][1];
      const auto image_width = image_widths[batch_index];
      const auto image_height = image_heights[batch_index];

      auto response = p.process(output_fine, output_coarse, output_det,
                                batch_index, image_width, image_height);
      auto text = dump_msg(response);
      {
        Response response_expect;
        auto r1 = google::protobuf::util::JsonStringToMessage(
            expect_response_text, &response_expect);
        EXPECT_EQ(r1.ok(), true);
        EXPECT_EQ(
            diff_protobuf_msg_with_precision(response, response_expect, 1e4),
            "");
      }
    }
  }
}

TEST(TerrorMixup, vector_marshal_to_string) {
  vector<float> xx{1, 3.1, 2.5, -1.1};
  string r;
  vector_marshal_to_string(xx, r);
  vector<float> xx2;
  string_unmarshal_to_vector(r, xx2);
  CHECK_EQ(xx, xx2);
}

TEST(TerrorMixup, TronTest) {
  PredictorContext ctx;

  inference::CreateParams create_params;
  const int batch_size = 4;
  {
    create_params.set_batch_size(batch_size);

    std::vector<std::string> model_names = {
        model_dir + "coarse_deploy.prototxt",
        model_dir + "coarse_weight.caffemodel",
        model_dir + "det_labels.csv",
        model_dir + "fine_deploy.prototxt",
        model_dir + "fine_weight.caffemodel",
        model_dir + "coarse_labels.csv",
        model_dir + "det_deploy.prototxt",
        model_dir + "det_weight.caffemodel",
        model_dir + "fine_labels.csv"};
    for (auto name : model_names) {
      string content = read_bin_file_to_string(name);
      auto *_model = create_params.add_model_files();
      _model->set_name(name);
      _model->set_body(content);
    }

    int gpu_id = 0;
    {
      const string s = getEnvVar("GPU_INDEX");
      if (!s.empty()) {
        LOG(INFO) << "use gpu:" << std::stoi(s);
        gpu_id = std::stoi(s);
      }
    }
    const std::string custom_params =
        boost::str(boost::format(R"({"gpu_id":%d})") % gpu_id);
    create_params.set_custom_params(custom_params);
  }
  auto _ = gsl::finally([&] { QTPredFree(ctx); });
  const auto create_params_str = create_params.SerializeAsString();
  auto code =
      QTPredCreate(create_params_str.data(), create_params_str.size(), &ctx);
  EXPECT_EQ(code, 0) << QTGetLastError();

  PredictorHandle handle;
  code = QTPredHandle(ctx, create_params_str.data(), create_params_str.size(),
                      &handle);
  EXPECT_EQ(code, 0) << QTGetLastError();

  const auto tsv =
      csv_parse(read_bin_file_to_string(
                    "../../res/testdata/image/serving/terror-mixup/"
                    "terror-mixup-201811211548/set20181108/201811211548.tsv"),
                "\t");
  for (std::size_t tsv_index = 0; tsv_index < tsv.size();
       tsv_index += batch_size) {
    _DEBUG_PRINT(tsv_index);
    vector<vector<string>> tsv_lines;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      std::size_t index = tsv_index + batch_index;
      if (index >= tsv.size()) {
        index = tsv.size() - 1;
      }
      tsv_lines.push_back(tsv[index]);
    }

    inference::InferenceRequests inference_requests;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      auto request = inference_requests.add_requests();
      const auto filename = tsv_lines.at(batch_index)[0];
      const string image_content = read_bin_file_to_string(
          "../../res/testdata/image/serving/terror-mixup/set20181108/" +
          filename);
      request->mutable_data()->set_body(image_content);
    }
    const auto inference_requests_str = inference_requests.SerializeAsString();
    void *inference_responses_data = nullptr;
    int inference_responses_size = 0;
    code = QTPredInference(
        handle, inference_requests_str.data(), inference_requests_str.size(),
        &inference_responses_data, &inference_responses_size);
    EXPECT_EQ(code, 0) << QTGetLastError();

    inference::InferenceResponses inference_responses;
    inference_responses.ParseFromArray(
        static_cast<char *>(inference_responses_data),
        inference_responses_size);

    for (int batch_index = 0;
         batch_index < inference_responses.responses_size(); ++batch_index) {
      auto resp = inference_responses.responses(batch_index);
      const auto expect_response_text = tsv_lines[batch_index][1];
      // LOG(INFO) << batch_index << " " << resp.result() << "\n"
      //           << expect_response_text;

      Response response_expect, response_actual;
      auto r = google::protobuf::util::JsonStringToMessage(expect_response_text,
                                                           &response_expect);
      EXPECT_EQ(r.ok(), true);

      r = google::protobuf::util::JsonStringToMessage(resp.result(),
                                                      &response_actual);
      EXPECT_EQ(r.ok(), true);

      EXPECT_EQ(diff_protobuf_msg_with_precision(response_actual,
                                                 response_expect, 1e6),
                "");
    }
  }
}

}  // namespace terror_mixup
}  // namespace tron
