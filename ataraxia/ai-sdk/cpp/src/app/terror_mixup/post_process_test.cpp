#include "post_process.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gmock/gmock.h"
#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"

namespace tron {
namespace terror_mixup {
using std::string;
using std::vector;

void assert_pts_equal(const int a[4][2], const int b[4][2]) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(a[i][j], b[i][j]);
    }
  }
}

// 完整测试，验证中间步骤
TEST(TerrorMixup, Process) {
  post_process_param param;
  {
    param.fine_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "fine_labels.csv");
    param.det_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/det_labels.csv");
    param.coarse_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "coarse_labels.csv");
    param.percent_fine = 0.6;
    param.percent_coarse = 0.4;
    param.batch_size = 1;
  }
  const PostProcess p(param);
  const vector<float> output_fine = {
      5.63378337e-11, 1.57534552e-04, 9.99449313e-01, 2.31584883e-04,
      1.38160727e-09, 3.85873511e-09, 3.30084693e-09, 2.49430355e-11,
      5.09967180e-10, 4.13394030e-09, 2.39331222e-09, 1.21484889e-09,
      9.94563720e-09, 2.42177940e-08, 1.09743512e-08, 1.05008851e-07,
      3.56382901e-09, 1.14866935e-10, 1.51838382e-08, 1.56070309e-04,
      9.13517109e-11, 1.59758429e-08, 5.88663840e-10, 2.42761339e-10,
      7.17360365e-07, 7.97797473e-09, 4.79447548e-09, 1.96872463e-07,
      1.40914655e-10, 9.30920563e-10, 3.11110182e-09, 6.61439259e-08,
      5.27027831e-11, 4.29949750e-06, 3.71829212e-09, 2.19135543e-10,
      4.25002394e-11, 1.81169186e-12, 6.44878421e-12, 1.12118700e-11,
      3.52286126e-12, 3.20512200e-10, 3.24303251e-10, 7.24519333e-10,
      1.02774400e-09, 6.97933517e-11, 4.51252324e-09, 9.29554003e-11};
  const vector<float> output_coarse = {
      5.6684927e-17, 1.0000000e+00, 6.5655777e-21, 2.5925641e-24,
      2.0746421e-17, 2.6511504e-17, 2.1047036e-23};
  vector<float> output_det(1 * 1 * 500 * 7);
  {
    const vector<float> output_det_t = {
        0.0000000e+00, 1.0000000e+01, 9.9532557e-01, 2.3226261e-02,
        3.1016469e-03, 8.7361497e-01, 5.7956433e-01};
    std::copy(output_det_t.begin(), output_det_t.end(), output_det.begin());
  }
  const int image_width = 800;
  const int image_height = 533;
  const int batch_index = 0;
  {
    // 检查参数
    assert_output_shape(output_fine, output_coarse, output_det, batch_index,
                        p.batch_size);
    vector<float> cls_result = cls_post_eval(
        output_coarse, output_fine, p.coarse_label_to_fine_label,
        p.fine_labels_v, batch_index, p.percent_fine, p.percent_coarse);
    {
      vector<float> cls_result_expect = {
          5.3208e-11, 0.000148783, 0.999505, 0.000218719, 1.30485e-09,
          3.64436e-09, 3.11747e-09, 2.35573e-11, 2.88981e-10, 3.90428e-09,
          2.26035e-09, 1.14736e-09, 9.3931e-09, 1.37234e-08, 6.2188e-09,
          5.9505e-08, 3.36584e-09, 1.08485e-10, 1.43403e-08, 0.0001474,
          8.62766e-11, 1.50883e-08, 5.5596e-10, 2.29275e-10, 6.77507e-07,
          7.53475e-09, 4.52812e-09, 1.85935e-07, 1.33086e-10, 5.27522e-10,
          2.93826e-09, 6.24693e-08, 4.97749e-11, 2.43638e-06, 3.51172e-09,
          2.06961e-10, 4.01391e-11, 1.71104e-12, 6.09052e-12, 1.0589e-11,
          3.32715e-12, 3.02706e-10, 3.06286e-10, 6.84268e-10, 9.70647e-10,
          6.59159e-11, 4.26183e-09, 8.77912e-11};
      EXPECT_THAT(cls_result_expect,
                  testing::Pointwise(testing::FloatNear(1e-5), cls_result));
    }
    auto det_results = det_post_eval(image_width, image_height, output_det,
                                     p.det_labels_v, batch_index);
    {
      EXPECT_EQ(det_results.size(), 1U);
      EXPECT_EQ(det_results[0].index, 10);
      EXPECT_EQ(det_results[0].class_name, "smoke");
      EXPECT_THAT(det_results[0].score,
                  testing::FloatNear(1e-5, 0.9953255653381348));
      int pts[4][2] = {{18, 1}, {698, 1}, {698, 308}, {18, 308}};
      assert_pts_equal(pts, det_results[0].pts);
    }
    auto cls_result2 =
        cls_merge_det(p.det_labels_v, p.fine_labels_v, det_results, cls_result);
    {
      vector<float> cls_result_expect2 = {
          5.3208e-11, 0.000148783, 0.999505, 0.000218719, 1.30485e-09,
          3.64436e-09, 3.11747e-09, 2.35573e-11, 2.88981e-10, 3.90428e-09,
          2.26035e-09, 1.14736e-09, 9.3931e-09, 1.37234e-08, 6.2188e-09,
          5.9505e-08, 3.36584e-09, 1.08485e-10, 1.43403e-08, 0.0001474,
          8.62766e-11, 1.50883e-08, 5.5596e-10, 2.29275e-10, 6.77507e-07,
          7.53475e-09, 4.52812e-09, 1.85935e-07, 1.33086e-10, 5.27522e-10,
          2.93826e-09, 6.24693e-08, 4.97749e-11, 2.43638e-06, 3.51172e-09,
          2.06961e-10, 4.01391e-11, 1.71104e-12, 6.09052e-12, 1.0589e-11,
          3.32715e-12, 3.02706e-10, 3.06286e-10, 6.84268e-10, 9.70647e-10,
          6.59159e-11, 4.26183e-09, 8.77912e-11};
      EXPECT_THAT(cls_result_expect2,
                  testing::Pointwise(testing::FloatNear(1e-5), cls_result2));
    }
  }

  const auto response = p.process(output_fine, output_coarse, output_det,
                                  batch_index, image_width, image_height);

  {
    auto response_sorted(response);
    std::sort(response_sorted.mutable_confidences()->begin(),
              response_sorted.mutable_confidences()->end(),
              [](auto &a, auto &b) { return a.index() < b.index(); });

    Response response_expect;
    {
      auto json = R"(
{"confidences": [{"index": 0, "score": 5.3207954053527094e-11, "class": "bloodiness"}, {"index": 2, "score": 0.9995043814182281, "class": "normal"}, {"index": 4, "score": 1.3048513134192614e-09, "class": "self_burning"}, {"index": 5, "score": 3.6443609364845817e-09, "class": "beheaded"}, {"index": 8, "score": 2.8898140177110185e-10, "class": "march_crowed"}, {"index": 9, "score": 3.904276951121548e-09, "class": "fight_police"}, {"index": 10, "score": 2.2603504253500737e-09, "class": "fight_person"}, {"index": 11, "score": 1.1473572832561747e-09, "class": "special_characters"}, {"index": 15, "score": 5.950501581026275e-08, "class": "anime_bloodiness"}, {"index": 17, "score": 1.0848543899630611e-10, "class": "special_clothing"}], "checkpoint": "endpoint"}
  )";
      auto r1 =
          google::protobuf::util::JsonStringToMessage(json, &response_expect);
      EXPECT_EQ(r1.ok(), true);
    }
    EXPECT_EQ(
        diff_protobuf_msg_with_precision(response_sorted, response_expect), "");
  }
}

// 完整测试2
TEST(TerrorMixup, Process2) {
  post_process_param param;
  {
    param.fine_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "fine_labels.csv");
    param.det_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/det_labels.csv");
    param.coarse_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "coarse_labels.csv");
    param.percent_fine = 0.6;
    param.percent_coarse = 0.4;
    param.batch_size = 2;
  }
  const PostProcess p(param);
  const vector<float> output_fine = {
      2.2666086e-11, 1.5243076e-12, 3.4646496e-13, 3.2974207e-13,
      8.848023e-12, 1.7746704e-13, 6.856971e-11, 2.7911245e-12,
      5.804003e-12, 2.4890807e-13, 2.2935111e-14, 4.2521392e-10,
      1.8248597e-11, 5.1892386e-11, 1.8486468e-11, 1.5476058e-09,
      1.2175126e-10, 5.06373e-13, 1.5613273e-09, 2.8194827e-12,
      2.8674113e-10, 2.2502107e-10, 1.5996416e-11, 2.1578958e-12,
      1.34939135e-11, 9.513001e-13, 1.0669212e-11, 9.814157e-12,
      2.3559192e-06, 1.4336e-08, 4.556062e-10, 8.4089247e-10,
      3.7468853e-10, 2.587504e-13, 1.9241906e-09, 5.737708e-12,
      1.4685471e-14, 8.014774e-12, 3.7109116e-10, 2.4451326e-12,
      3.865006e-10, 4.820502e-10, 2.1613364e-11, 1.3355676e-10,
      2.3696797e-14, 0.9999976, 5.033904e-10, 1.2563415e-08,
      2.0761901e-07, 3.1029558e-06, 1.339254e-06, 7.4611125e-07,
      1.2659709e-06, 8.0984125e-07, 1.14126e-06, 2.126639e-08,
      8.725304e-07, 1.4510324e-07, 2.2522156e-07, 0.0005602117,
      1.16401425e-05, 4.8562373e-07, 1.1787334e-06, 5.604505e-07,
      1.0366526e-06, 3.2221326e-06, 6.6759395e-07, 9.1481127e-07,
      4.4158583e-06, 0.0023857194, 5.858283e-07, 2.75645e-07,
      5.8188164e-07, 1.3009892e-06, 4.245258e-07, 4.886733e-07,
      2.841623e-05, 0.00011548164, 5.8940304e-07, 1.8264087e-06,
      1.7920833e-05, 7.0197802e-06, 3.4234097e-06, 2.4932797e-07,
      2.8520566e-07, 8.630941e-07, 2.8408667e-07, 7.75988e-08,
      9.788105e-07, 1.1701792e-05, 1.1816372e-06, 2.0290024e-07,
      2.1812988e-07, 0.002607078, 4.7978672e-05, 0.9941707};  // NOLINT
  const vector<float> output_coarse = {
      3.4160487e-11, 9.449849e-13, 1.3700998e-11, 1.0916843e-12, 1.3591224e-13,
      6.4559163e-06, 0.99999356, 1.7542522e-09, 7.2075466e-07, 1.8345482e-06,
      6.9222833e-10, 1.6773284e-10, 0.9999974, 1.1366915e-07};  // NOLINT
  vector<float> output_det(1 * 1 * 500 * 7);
  {
    const vector<float> output_det_t = {
        0.0, 4.0, 0.7051198, 0.20843512, 0.1899412,
        0.90799624, 0.37083337, 0.0, 4.0, 0.21157393,
        0.54679716, 0.63407356, 0.7109709, 0.7159658, 0.0,
        4.0, 0.12113853, 0.23824471, 0.58664316, 0.9021726,
        0.71512944, 0.0, 4.0, 0.10641196, 0.08802551,
        0.7148315, 0.4806239, 0.75374997, 0.0, 4.0,
        0.10592297, 0.06618157, 0.68008685, 0.7042264, 0.7709155,
        0.0, 12.0, 0.22294556, 0.22424257, 0.17720571,
        0.89271116, 0.36883733};  // NOLINT
    std::copy(output_det_t.begin(), output_det_t.end(), output_det.begin());
  }
  const int image_width = 1200;
  const int image_height = 360;
  const int batch_index = 0;
  const auto response = p.process(output_fine, output_coarse, output_det,
                                  batch_index, image_width, image_height);
  {
    auto response_sorted(response);

    Response response_expect;
    {
      auto json = R"(
{"confidences": [{"index": 0, "score": 2.1406859020751186e-11, "class": "bloodiness"}, {"index": 45, "score": 0.9999939918518067, "class": "normal"}, {"index": 4, "score": 8.356466174894469e-12, "class": "self_burning"}, {"index": 6, "score": 6.476028394097887e-11, "class": "beheaded"}, {"index": 8, "score": 3.7013491471242655e-12, "class": "march_crowed"}, {"index": 9, "score": 2.350798456720801e-13, "class": "fight_police"}, {"index": 10, "score": 2.1660938273204428e-14, "class": "fight_person"}, {"index": 11, "score": 2.4613048696001423e-10, "class": "special_characters"}, {"index": 15, "score": 1.4616277200908536e-09, "class": "anime_bloodiness"}, {"index": 17, "score": 4.782411596608505e-13, "class": "special_clothing"}], "checkpoint": "terror-detect"}
  )";
      auto r1 =
          google::protobuf::util::JsonStringToMessage(json, &response_expect);
      EXPECT_EQ(r1.ok(), true);
    }
    EXPECT_EQ(
        diff_protobuf_msg_with_precision(response_sorted, response_expect), "");
  }
}

// 完整测试3, 跑到了 use score_map 逻辑
TEST(TerrorMixup, Process3) {
  post_process_param param;
  {
    param.fine_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "fine_labels.csv");
    param.det_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/det_labels.csv");
    param.coarse_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "coarse_labels.csv");
    param.percent_fine = 0.6;
    param.percent_coarse = 0.4;
    param.batch_size = 2;
  }
  const PostProcess p(param);
  const vector<float> output_fine = {
      0.99983454, 1.2342319e-09, 2.7915825e-10, 2.158353e-10,
      6.9504753e-09, 9.989388e-08, 1.7093963e-07, 4.1181223e-09,
      2.9896569e-10, 6.033382e-09, 1.6075315e-05, 1.9612804e-08,
      3.948785e-11, 4.5676474e-10, 3.433106e-08, 3.174064e-07,
      1.2282078e-10, 9.608707e-10, 6.4257426e-07, 9.543283e-10,
      1.2839624e-11, 2.5064515e-14, 1.7581532e-09, 5.460076e-07,
      3.4244196e-05, 3.6205333e-06, 3.8705422e-10, 3.422896e-10,
      2.5195288e-09, 5.103846e-10, 5.185051e-10, 1.0446653e-10,
      1.9724342e-11, 2.205844e-10, 2.8973629e-08, 2.3965366e-10,
      5.4788357e-10, 6.106245e-10, 1.2882181e-08, 1.0318887e-09,
      5.326271e-12, 1.0025985e-10, 1.2067654e-10, 4.834274e-09,
      5.4460575e-10, 5.4801346e-09, 0.00010956157, 1.4217637e-12,
      2.0761901e-07, 3.1029558e-06, 1.339254e-06, 7.4611125e-07,
      1.2659709e-06, 8.0984125e-07, 1.14126e-06, 2.126639e-08,
      8.725304e-07, 1.4510324e-07, 2.2522156e-07, 0.0005602117,
      1.16401425e-05, 4.8562373e-07, 1.1787334e-06, 5.604505e-07,
      1.0366526e-06, 3.2221326e-06, 6.6759395e-07, 9.1481127e-07,
      4.4158583e-06, 0.0023857194, 5.858283e-07, 2.75645e-07,
      5.8188164e-07, 1.3009892e-06, 4.245258e-07, 4.886733e-07,
      2.841623e-05, 0.00011548164, 5.8940304e-07, 1.8264087e-06,
      1.7920833e-05, 7.0197802e-06, 3.4234097e-06, 2.4932797e-07,
      2.8520566e-07, 8.630941e-07, 2.8408667e-07, 7.75988e-08,
      9.788105e-07, 1.1701792e-05, 1.1816372e-06, 2.0290024e-07,
      2.1812988e-07, 0.002607078, 4.7978672e-05, 0.9941707};  // NOLINT
  const vector<float> output_coarse = {
      1.0, 8.0944353e-19, 1.5910205e-09, 9.814169e-18, 1.4665341e-08,
      1.0340835e-12, 2.9807994e-19, 1.7542522e-09, 7.2075466e-07, 1.8345482e-06,
      6.9222833e-10, 1.6773284e-10, 0.9999974, 1.1366915e-07};  // NOLINT
  vector<float> output_det(1 * 1 * 500 * 7);
  {
    const vector<float> output_det_t = {
        0.0, 4.0, 0.17151707, 0.115119874, 0.5094049,
        0.90810376, 0.96138847, 0.0, 6.0, 0.1721489,
        0.12587833, 0.47299215, 0.89883614, 0.97410476, 0.0,
        6.0, 0.107770726, 0.48589313, 0.51896936, 0.74464846,
        0.7647318, 0.0, 12.0, 0.1914344, 0.18262532,
        0.55744743, 0.8882663, 0.97913074};  // NOLINT
    std::copy(output_det_t.begin(), output_det_t.end(), output_det.begin());
  }
  const int image_width = 405;
  const int image_height = 270;
  const int batch_index = 0;
  const auto response = p.process(output_fine, output_coarse, output_det,
                                  batch_index, image_width, image_height);
  {
    auto response_sorted(response);

    Response response_expect;
    {
      auto json = R"(
{"confidences": [{"index": 0, "score": 0.84, "class": "bloodiness"}, {"index": 46, "score": 6.208489202999319e-05, "class": "normal"}, {"index": 4, "score": 3.938602673938638e-09, "class": "self_burning"}, {"index": 6, "score": 9.746684178511976e-08, "class": "beheaded"}, {"index": 8, "score": 2.8235648078892933e-10, "class": "march_crowed"}, {"index": 9, "score": 5.698193916714444e-09, "class": "fight_police"}, {"index": 10, "score": 9.114885528897313e-06, "class": "fight_person"}, {"index": 11, "score": 1.852320361292767e-08, "class": "special_characters"}, {"index": 15, "score": 2.9977271430602235e-07, "class": "anime_bloodiness"}, {"index": 17, "score": 9.074890098640744e-10, "class": "special_clothing"}], "checkpoint": "terror-detect"}
  )";  // NOLINT
      auto r1 =
          google::protobuf::util::JsonStringToMessage(json, &response_expect);
      EXPECT_EQ(r1.ok(), true);
    }
    EXPECT_EQ(
        diff_protobuf_msg_with_precision(response_sorted, response_expect), "");
  }
}

// 完整测试4, 跑到了 use score_map 逻辑，batch_index不是0
TEST(TerrorMixup, Process4) {
  post_process_param param;
  {
    param.fine_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "fine_labels.csv");
    param.det_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/det_labels.csv");
    param.coarse_labels = read_bin_file_to_string(
        "../../res/model/ava-terror-mixup_terror-mixup-1121-11/"
        "coarse_labels.csv");
    param.percent_fine = 0.6;
    param.percent_coarse = 0.4;
    param.batch_size = 2;
  }
  const PostProcess p(param);
  vector<float> output_fine(48 * 2);
  {
    const vector<float> output_fine_t = {
        0.99983454, 1.2342319e-09, 2.7915825e-10, 2.158353e-10,
        6.9504753e-09, 9.989388e-08, 1.7093963e-07, 4.1181223e-09,
        2.9896569e-10, 6.033382e-09, 1.6075315e-05, 1.9612804e-08,
        3.948785e-11, 4.5676474e-10, 3.433106e-08, 3.174064e-07,
        1.2282078e-10, 9.608707e-10, 6.4257426e-07, 9.543283e-10,
        1.2839624e-11, 2.5064515e-14, 1.7581532e-09, 5.460076e-07,
        3.4244196e-05, 3.6205333e-06, 3.8705422e-10, 3.422896e-10,
        2.5195288e-09, 5.103846e-10, 5.185051e-10, 1.0446653e-10,
        1.9724342e-11, 2.205844e-10, 2.8973629e-08, 2.3965366e-10,
        5.4788357e-10, 6.106245e-10, 1.2882181e-08, 1.0318887e-09,
        5.326271e-12, 1.0025985e-10, 1.2067654e-10, 4.834274e-09,
        5.4460575e-10, 5.4801346e-09, 0.00010956157, 1.4217637e-12,
        2.0761901e-07, 3.1029558e-06, 1.339254e-06, 7.4611125e-07,
        1.2659709e-06, 8.0984125e-07, 1.14126e-06, 2.126639e-08,
        8.725304e-07, 1.4510324e-07, 2.2522156e-07, 0.0005602117,
        1.16401425e-05, 4.8562373e-07, 1.1787334e-06, 5.604505e-07,
        1.0366526e-06, 3.2221326e-06, 6.6759395e-07, 9.1481127e-07,
        4.4158583e-06, 0.0023857194, 5.858283e-07, 2.75645e-07,
        5.8188164e-07, 1.3009892e-06, 4.245258e-07, 4.886733e-07,
        2.841623e-05, 0.00011548164, 5.8940304e-07, 1.8264087e-06,
        1.7920833e-05, 7.0197802e-06, 3.4234097e-06, 2.4932797e-07,
        2.8520566e-07, 8.630941e-07, 2.8408667e-07, 7.75988e-08,
        9.788105e-07, 1.1701792e-05, 1.1816372e-06, 2.0290024e-07,
        2.1812988e-07, 0.002607078, 4.7978672e-05, 0.9941707};  // NOLINT
    std::copy(output_fine_t.begin(), output_fine_t.begin() + 48,
              output_fine.begin() + 48);
  }
  vector<float> output_coarse(7 * 2);
  {
    const vector<float> output_coarse_t = {
        1.0, 8.0944353e-19, 1.5910205e-09, 9.814169e-18,
        1.4665341e-08, 1.0340835e-12, 2.9807994e-19, 1.7542522e-09,
        7.2075466e-07, 1.8345482e-06, 6.9222833e-10, 1.6773284e-10,
        0.9999974, 1.1366915e-07};  // NOLINT
    std::copy(output_coarse_t.begin(), output_coarse_t.begin() + 7,
              output_coarse.begin() + 7);
  }
  vector<float> output_det(1 * 1 * 500 * 7);
  {
    vector<float> output_det_t = {
        0.0, 4.0, 0.17151707, 0.115119874, 0.5094049,
        0.90810376, 0.96138847, 0.0, 6.0, 0.1721489,
        0.12587833, 0.47299215, 0.89883614, 0.97410476, 0.0,
        6.0, 0.107770726, 0.48589313, 0.51896936, 0.74464846,
        0.7647318, 0.0, 12.0, 0.1914344, 0.18262532,
        0.55744743, 0.8882663, 0.97913074};  // NOLINT
    for (auto i_bbox = output_det_t.begin(); i_bbox < output_det_t.end();
         i_bbox += output_det_class_num) {
      int image_id = static_cast<int>(*(i_bbox));
      if (image_id == 0) {
        *i_bbox = 1;
      }
    }
    std::copy(output_det_t.begin(), output_det_t.end(), output_det.begin());
  }
  const int image_width = 405;
  const int image_height = 270;
  const int batch_index = 1;
  const auto response = p.process(output_fine, output_coarse, output_det,
                                  batch_index, image_width, image_height);
  {
    auto response_sorted(response);

    Response response_expect;
    {
      auto json = R"(
{"confidences": [{"index": 0, "score": 0.84, "class": "bloodiness"}, {"index": 46, "score": 6.208489202999319e-05, "class": "normal"}, {"index": 4, "score": 3.938602673938638e-09, "class": "self_burning"}, {"index": 6, "score": 9.746684178511976e-08, "class": "beheaded"}, {"index": 8, "score": 2.8235648078892933e-10, "class": "march_crowed"}, {"index": 9, "score": 5.698193916714444e-09, "class": "fight_police"}, {"index": 10, "score": 9.114885528897313e-06, "class": "fight_person"}, {"index": 11, "score": 1.852320361292767e-08, "class": "special_characters"}, {"index": 15, "score": 2.9977271430602235e-07, "class": "anime_bloodiness"}, {"index": 17, "score": 9.074890098640744e-10, "class": "special_clothing"}], "checkpoint": "terror-detect"}
  )";
      auto r1 =
          google::protobuf::util::JsonStringToMessage(json, &response_expect);
      EXPECT_EQ(r1.ok(), true);
    }
    EXPECT_EQ(
        diff_protobuf_msg_with_precision(response_sorted, response_expect), "");
  }
}

}  // namespace terror_mixup
}  // namespace tron
