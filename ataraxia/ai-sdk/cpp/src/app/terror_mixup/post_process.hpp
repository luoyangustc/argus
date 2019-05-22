#ifndef TRON_TERROR_MIXUP_POST_PROCESS  // NOLINT
#define TRON_TERROR_MIXUP_POST_PROCESS

#include <string>
#include <utils.hpp>
#include <vector>
#include "./forward.pb.h"
#include "./utils.hpp"

namespace tron {
namespace terror_mixup {

using std::string;
using std::vector;

struct post_process_param {
  string fine_labels;
  string det_labels;
  string coarse_labels;
  float percent_fine;
  float percent_coarse;
  int batch_size;
};

inline std::ostream &operator<<(std::ostream &o, const post_process_param &p) {
  return o << "post_process_param{"
           << "fine_labels:" << p.fine_labels << ", "
           << "det_labels:" << p.det_labels << ", "
           << "coarse_labels:" << p.coarse_labels << ", "
           << "percent_fine:" << p.percent_fine << ", "
           << "percent_coarse:" << p.percent_coarse << ", "
           << "batch_size:" << p.batch_size
           << "};";
}

struct fine_labels_column {
  int index = -1;
  string class_name;
  float model_threshold = -1;
  float serving_threshold = -1;
};

inline std::ostream &operator<<(std::ostream &o, const fine_labels_column &p) {
  return o << "fine_labels_column{"
           << "index:" << p.index << ", "
           << "class_name:" << p.class_name << ", "
           << "model_threshold:" << p.model_threshold << ", "
           << "serving_threshold:" << p.serving_threshold
           << "};";
}

struct det_labels_column {
  int index = -1;
  string class_name;
  float threshold = -1;
  string clsNeed;
  bool isClsNeed = false;
  vector<int> cls_index;
  bool isPredet = false;
};

inline std::ostream &operator<<(std::ostream &o, const det_labels_column &p) {
  return o << "det_labels_column{"
           << "index:" << p.index << ", "
           << "class_name:" << p.class_name << ", "
           << "threshold:" << p.threshold << ", "
           << "clsNeed:" << p.clsNeed << ", "
           << "isClsNeed:" << p.isClsNeed << ", "
           << "cls_index:" << p.cls_index << ", "
           << "isPredet:" << p.isPredet
           << "};";
}

struct det_image_result {
  int index = -1;
  float score = -1;
  int pts[4][2];
  string class_name;
};

inline std::ostream &operator<<(std::ostream &o, const det_image_result &p) {
  return o << "det_image_result{"
           << "index:" << p.index << ", "
           << "score:" << p.score << ", "
           << "pts:" << p.pts << ", "
           << "class_name:" << p.class_name
           << "};";
}

const auto output_fine_shape =
    NetworkShape("output_fine_shape",
                 {-1, 48, 1, 1},
                 NetworkShapeType::Normal);
const auto output_coarse_shape =
    NetworkShape("output_coarse_shape",
                 {-1, 7, 1, 1},
                 NetworkShapeType::Normal);
const int output_det_class_num = 7;
const auto output_det_shape =
    NetworkShape("output_det_shape",
                 {-1, 1, 500, output_det_class_num},
                 NetworkShapeType::BatchSizeUnrelated);

const auto input_fine_shape =
    NetworkShape("input_fine_shape",
                 {-1, 3, 225, 225},
                 NetworkShapeType::Normal);
const auto input_coarse_shape =
    NetworkShape("input_coarse_shape",
                 {-1, 3, 225, 225},
                 NetworkShapeType::Normal);
const int input_det_class_num = 7;
const auto input_det_shape =
    NetworkShape("input_det_shape",
                 {-1, 3, 320, 320},
                 NetworkShapeType::Normal);

int get_max_idx(vector<float>::const_iterator array, vector<int> index_list);

void assert_output_shape(
    const vector<float> &output_fine,
    const vector<float> &output_coarse,
    const vector<float> &output_det,
    const int batch_index,
    const int batch_size);

vector<vector<int>> get_coarse_label_to_fine_label(
    const csv_fields &coarse_labels);

vector<fine_labels_column> get_fine_labels_v(const csv_fields &fine_labels);

vector<float> cls_post_eval(
    const vector<float> &output_coarse,
    const vector<float> &output_fine,
    const vector<vector<int>> &coarse_label_to_fine_label,
    const vector<fine_labels_column> &fine_labels_v,
    const int batch_index,
    const float percent_fine,
    const float percent_coarse);

vector<float> cls_merge_det(
    const vector<det_labels_column> &det_labels_v,
    const vector<fine_labels_column> &fine_labels_v,
    const vector<det_image_result> &det_results,
    const vector<float> &cls_result);

Response merge_confidences(
    const vector<fine_labels_column> &fine_labels_v,
    const vector<float> &cls_result);

string get_checkpoint(
    const vector<det_labels_column> &det_labels_v,
    const vector<det_image_result> &det_results);

vector<det_labels_column> get_det_labels_v(const csv_fields &det_labels);

vector<det_image_result> det_post_eval(
    const int image_width,
    const int image_height,
    const vector<float> &output,
    const vector<det_labels_column> &det_labels_v,
    const int batch_index);

class PostProcess {
 public:
  int batch_size;
  float percent_fine;
  float percent_coarse;
  vector<fine_labels_column> fine_labels_v;
  vector<det_labels_column> det_labels_v;
  vector<vector<int>> coarse_label_to_fine_label;

  explicit PostProcess(const post_process_param &param) {
    percent_fine = param.percent_fine;
    percent_coarse = param.percent_coarse;
    batch_size = param.batch_size;

    CHECK_GT(batch_size, 0);
    CHECK_GT(percent_fine, 0);
    CHECK_GT(percent_coarse, 0);

    const auto fine_labels = csv_parse(param.fine_labels);
    const auto det_labels = csv_parse(param.det_labels);
    const auto coarse_labels = csv_parse(param.coarse_labels);

    coarse_label_to_fine_label = get_coarse_label_to_fine_label(coarse_labels);
    fine_labels_v = get_fine_labels_v(fine_labels);
    det_labels_v = get_det_labels_v(det_labels);
  }
  Response process(const vector<float> &output_fine,
                   const vector<float> &output_coarse,
                   const vector<float> &output_det,
                   const int batch_index,
                   const int image_width, const int image_height) const {
    assert_output_shape(
        output_fine,
        output_coarse,
        output_det,
        batch_index,
        batch_size);
    const vector<float> cls_result = cls_post_eval(
        output_coarse,
        output_fine,
        coarse_label_to_fine_label,
        fine_labels_v,
        batch_index,
        percent_fine,
        percent_coarse);
    const auto det_results = det_post_eval(
        image_width,
        image_height,
        output_det,
        det_labels_v,
        batch_index);
    const auto cls_result2 = cls_merge_det(
        det_labels_v,
        fine_labels_v,
        det_results,
        cls_result);

    Response response = merge_confidences(fine_labels_v, cls_result2);
    response.set_checkpoint(get_checkpoint(det_labels_v, det_results));
    return response;
  }
};

}  // namespace terror_mixup
}  // namespace tron
#endif  // TRON_FACE_FEATURE_FORWARD_FEATURE_HPP NOLINT
