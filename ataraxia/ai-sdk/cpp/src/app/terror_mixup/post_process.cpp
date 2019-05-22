#include "post_process.hpp"
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "gsl/span"

namespace tron {
namespace terror_mixup {
using gsl::make_span;
using gsl::span;
using std::string;
using std::vector;

int get_max_idx(span<const float> array, span<const int> index_list) {
  auto max_idx = index_list[0];
  auto max_value = array[max_idx];
  for (auto i : index_list) {
    auto v = array[i];
    if (v > max_value) {
      max_idx = i;
      max_value = v;
    }
  }
  return max_idx;
}

void assert_output_shape(const vector<float> &output_fine,
                         const vector<float> &output_coarse,
                         const vector<float> &output_det, const int batch_index,
                         const int batch_size) {
  CHECK(batch_index < batch_size) << "batch_index < batch_size";
  output_fine_shape.assert_shape_match(output_fine, batch_size);
  output_coarse_shape.assert_shape_match(output_coarse, batch_size);
  output_det_shape.assert_shape_match(output_det, batch_size);
}

vector<vector<int>> get_coarse_label_to_fine_label(
    const csv_fields &coarse_labels) {
  vector<vector<int>> coarse_label_to_fine_label;
  coarse_label_to_fine_label.resize(coarse_labels.size());
  const auto csv_fields = coarse_labels;
  for (const auto &line_fields : csv_fields) {
    CHECK_EQ(line_fields.size(), 2) << "coarse_label col should 2";
    const auto coarse_label = std::stoi(line_fields[0]);
    const auto fine_label = std::stoi(line_fields[1]);
    coarse_label_to_fine_label.at(fine_label).push_back(coarse_label);
  }
  int max_index = 0;
  for (const auto &v : coarse_label_to_fine_label) {
    if (v.empty()) {
      break;
    }
    ++max_index;
  }
  coarse_label_to_fine_label.resize(max_index);
  return coarse_label_to_fine_label;
}

vector<fine_labels_column> get_fine_labels_v(const csv_fields &fine_labels) {
  vector<fine_labels_column> fine_labels_v;
  for (std::size_t i = 1; i < fine_labels.size(); ++i) {
    const auto row = fine_labels[i];
    fine_labels_column t;
    t.index = std::stoi(row[0]);
    t.class_name = row[1];
    t.model_threshold = std::stof(row[2]);
    t.serving_threshold = std::stof(row[3]);
    fine_labels_v.push_back(t);
  }
  for (std::size_t i = 0; i < fine_labels_v.size(); ++i) {
    CHECK_EQ(fine_labels_v[i].index, i);
  }
  CHECK_EQ(fine_labels_v[0].index, 0);
  return fine_labels_v;
}

vector<float> cls_post_eval(
    const vector<float> &output_coarse, const vector<float> &output_fine,
    const vector<vector<int>> &coarse_label_to_fine_label,
    const vector<fine_labels_column> &fine_labels_v, const int batch_index,
    const float percent_fine, const float percent_coarse) {
  vector<float> cls_result;
  const auto output_coarse_s =
      make_span(output_coarse)
          .subspan(output_coarse_shape.single_batch_size() * batch_index,
                   output_coarse_shape.single_batch_size());
  const auto output_fine_s =
      make_span(output_fine)
          .subspan(output_fine_shape.single_batch_size() * batch_index,
                   output_fine_shape.single_batch_size());

  vector<float> new_conf(output_fine_s.begin(), output_fine_s.end());
  for (std::size_t key = 0; key < coarse_label_to_fine_label.size(); ++key) {
    const auto index_list = coarse_label_to_fine_label[key];
    const auto max_idx = get_max_idx(output_fine_s, index_list);
    new_conf.at(max_idx) = output_coarse_s[key] * percent_coarse +
                           output_fine_s[max_idx] * percent_fine;
  }
  vector<float> score_map_conf(new_conf.size());
  for (std::size_t index = 0; index < new_conf.size(); ++index) {
    float score_map = 0;
    const auto score = new_conf[index];
    const auto model_threshold = fine_labels_v[index].model_threshold;
    const auto serving_threshold = fine_labels_v[index].serving_threshold;
    if (score < model_threshold) {
      score_map = score * serving_threshold / model_threshold;
    } else {
      score_map =
          1 - (1 - score) * (1 - serving_threshold) / (1 - model_threshold);
    }
    score_map_conf[index] = score_map;
  }
  return score_map_conf;
}

vector<float> cls_merge_det(const vector<det_labels_column> &det_labels_v,
                            const vector<fine_labels_column> &fine_labels_v,
                            const vector<det_image_result> &det_results,
                            const vector<float> &cls_result) {
  const int need_merge_det_idx = 1;
  const int det_find_lable_idx = 2;

  vector<int> ids(det_labels_v.size());
  for (const auto &det_label : det_labels_v) {
    if (det_label.isClsNeed) {
      ids.at(det_label.index) = need_merge_det_idx;
    }
  }
  for (const auto &det_result : det_results) {
    if (ids.at(det_result.index) == need_merge_det_idx) {
      ids.at(det_result.index) = det_find_lable_idx;
    }
  }

  vector<float> cls_result2(cls_result);
  for (std::size_t idx = 0; idx < ids.size(); ++idx) {
    if (ids[idx] != need_merge_det_idx) {
      continue;
    }
    for (const auto &cls_index : det_labels_v[idx].cls_index) {
      const auto serving_threshold = fine_labels_v[cls_index].serving_threshold;
      const auto label = fine_labels_v[cls_index].class_name;
      const auto score = cls_result[cls_index];
      if (score >= serving_threshold) {
        if (label != "normal") {
          auto score_map = serving_threshold - 0.01;
          cls_result2[cls_index] = score_map;
        }
      }
    }
  }
  return cls_result2;
}

Response merge_confidences(const vector<fine_labels_column> &fine_labels_v,
                           const vector<float> &cls_result) {
  Response response;
  vector<string> label_index_keys;
  vector<vector<int>> label_index_values;
  for (std::size_t key = 0; key < fine_labels_v.size(); ++key) {
    const auto label = fine_labels_v[key].class_name;
    auto pos =
        std::find(label_index_keys.begin(), label_index_keys.end(), label);
    if (pos == label_index_keys.end()) {
      label_index_keys.push_back(label);
      label_index_values.push_back(vector<int>{static_cast<int>(key)});
    } else {
      label_index_values.at(std::distance(label_index_keys.begin(), pos))
          .push_back(key);
    }
  }

  for (std::size_t i = 0; i < label_index_keys.size(); ++i) {
    const auto key = label_index_keys[i];
    const auto labels = label_index_values[i];
    float max_score = -1;
    for (const auto idx : labels) {
      auto score = cls_result.at(idx);
      if (score > max_score) {
        max_score = score;
      }
    }
    const auto index = get_max_idx(cls_result, labels);
    auto res = response.add_confidences();
    res->set_index(index);
    res->set_class_(key);
    res->set_score(max_score);
  }
  return response;
}

string get_checkpoint(const vector<det_labels_column> &det_labels_v,
                      const vector<det_image_result> &det_results) {
  auto sendToDetectModelFlag = 0;
  for (const auto &det_result : det_results) {
    const auto index = det_result.index;
    if (det_labels_v[index].isPredet) {
      sendToDetectModelFlag = 1;
    }
  }
  if (sendToDetectModelFlag == 1) {
    return "terror-detect";
  }
  return "endpoint";
}

vector<det_labels_column> get_det_labels_v(const csv_fields &det_labels) {
  // det_labels 的index从1开始
  vector<det_labels_column> det_labels_v;
  det_labels_v.resize(1);
  for (std::size_t i = 1; i < det_labels.size(); ++i) {
    const auto row = det_labels[i];
    det_labels_column t;
    t.index = std::stoi(row[0]);
    t.class_name = row[1];
    t.threshold = std::stof(row[2]);
    t.clsNeed = row[3];
    if (t.clsNeed.find("yes") != string::npos) {
      t.isClsNeed = true;
      const auto s = split_str(t.clsNeed, "_");
      for (std::size_t j = 1; j < s.size(); ++j) {
        t.cls_index.push_back(std::stoi(s[j]));
      }
    } else {
      t.isClsNeed = false;
    }
    if (t.clsNeed == "Predet") {
      t.isPredet = true;
    }
    det_labels_v.push_back(t);
  }
  CHECK_EQ(det_labels_v[0].index, -1);
  CHECK_EQ(det_labels_v[1].index, 1);
  return det_labels_v;
}

vector<det_image_result> det_post_eval(
    const int image_width, const int image_height, const vector<float> &output,
    const vector<det_labels_column> &det_labels_v, const int batch_index) {
  vector<det_image_result> image_results;
  for (auto i_bbox = output.begin(); i_bbox < output.end();
       i_bbox += output_det_class_num) {
    int image_id = static_cast<int>(*(i_bbox));
    if (image_id != batch_index) {
      continue;
    }
    const int class_index = static_cast<int>(*(i_bbox + 1));
    if (class_index < 1) {
      continue;
    }
    const auto score = *(i_bbox + 2);
    if (score < det_labels_v[class_index].threshold) {
      continue;
    }
    det_image_result t;
    t.class_name = det_labels_v[class_index].class_name;
    t.index = class_index;
    t.score = score;
    const float bbox[4] = {
        image_width * *(i_bbox + 3), image_height * *(i_bbox + 4),
        image_width * *(i_bbox + 5), image_height * *(i_bbox + 6)};
    const int xmin =
        static_cast<int>(bbox[0]) > 0 ? static_cast<int>(bbox[0]) : 0;
    const int ymin =
        static_cast<int>(bbox[1]) > 0 ? static_cast<int>(bbox[1]) : 0;
    const int xmax = static_cast<int>(bbox[2]) < image_width
                         ? static_cast<int>(bbox[2])
                         : image_width;
    const int ymax = static_cast<int>(bbox[3]) < image_height
                         ? static_cast<int>(bbox[3])
                         : image_height;
    t.pts[0][0] = xmin;
    t.pts[0][1] = ymin;

    t.pts[1][0] = xmax;
    t.pts[1][1] = ymin;

    t.pts[2][0] = xmax;
    t.pts[2][1] = ymax;

    t.pts[3][0] = xmin;
    t.pts[3][1] = ymax;
    image_results.push_back(t);
  }
  return image_results;
}

}  // namespace terror_mixup
}  // namespace tron
