#include "inference_fd.hpp"

#include "glog/logging.h"

#include "common/boxes.hpp"
#include "common/image.hpp"
#include "common/time.hpp"

namespace tron {
namespace fd {

template <typename T>
inline bool SortScorePairDescend(const std::pair<float, T> &pair1,
                                 const std::pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

inline void GetMaxScoreIndex(
    const VecFloat &scores, float threshold, int top_k,
    std::vector<std::pair<float, int>> *score_index_vec) {
  // Generate index score pairs.
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && static_cast<size_t>(top_k) < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

inline void ApplyNMSFast(const VecBoxF &bboxes, const VecFloat &scores,
                         float score_threshold, float nms_threshold, int top_k,
                         VecInt *indices) {
  // Sanity check.
  CHECK_EQ(bboxes.size(), scores.size());

  // Get top_k scores (with corresponding indices).
  std::vector<std::pair<float, int>> score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

  // Do nms.
  indices->clear();
  while (!score_index_vec.empty()) {
    int idx = score_index_vec.front().second;
    bool keep = true;
    for (auto id : *indices) {
      if (keep) {
        float overlap = Boxes::IoU(bboxes[idx], bboxes[id]);
        keep = overlap <= nms_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
  }
}

void FDInference::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  engine_ = engine;
}

void FDInference::Predict(const std::vector<FDRequest> &ins,
                          std::vector<VecBoxF> *outs) {
  in_data_.resize(ins.size() * in_num_);

  auto t1 = Time();
  std::vector<tensord::core::NetIn<float>> requests;

  for (std::size_t i = 0; i < ins.size(); ++i) {
    ConvertData(ins[i].im_mat, in_data_.data() + i * in_num_,
                ins[i].roi, in_c_, in_h_, in_w_);
    tensord::core::NetIn<float> request;
    request.names.push_back("data");
    request.datas.emplace_back(in_data_.begin() + i * in_num_,
                               in_data_.begin() + i * in_num_ + in_num_);
    requests.push_back(request);
  }

  std::vector<tensord::core::NetOut<float>> responses;
  auto t2 = Time();
  engine_->Predict(requests, &responses);
  auto t3 = Time();

  CHECK_EQ(responses.size(), ins.size());
  outs->clear();

  for (std::size_t i = 0; i < responses.size(); i++) {
    auto resp = responses[i];

    auto loc_data = resp.GetByName("odm_loc");
    auto conf_data = resp.GetByName("odm_conf_flatten");

    VecBoxF bboxes;
    for (std::size_t j = 0; j < loc_data.size(); j += 4) {
      bboxes.emplace_back(loc_data[j + 0],
                          loc_data[j + 1],
                          loc_data[j + 2],
                          loc_data[j + 3]);
    }
    VecFloat scores;
    for (std::size_t j = 0; j < conf_data.size(); j += 2) {
      scores.push_back(conf_data[j + 1]);
    }

    VecInt indices;
    ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_,
                 top_k_, &indices);
    if (keep_top_k_ > -1 && static_cast<int>(indices.size()) > keep_top_k_) {
      std::vector<std::pair<float, int>> score_index_pairs;
      for (const auto &idx : indices) {
        score_index_pairs.emplace_back(scores[idx], idx);
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<int>);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      VecInt new_indices;
      for (const auto &score_index_pair : score_index_pairs) {
        new_indices.push_back(score_index_pair.second);
      }
      indices = new_indices;
    }

    float height = ins[i].roi.h, width = ins[i].roi.w;
    VecBoxF boxes;
    for (const auto &idx : indices) {
      float score = scores[idx];
      if (score > threshold_) {
        BoxF clip_bbox;
        Boxes::Clip(bboxes[idx], &clip_bbox, 0.f, 1.f);
        clip_bbox.xmin = clip_bbox.xmin * width;
        clip_bbox.ymin = clip_bbox.ymin * height;
        clip_bbox.xmax = clip_bbox.xmax * width;
        clip_bbox.ymax = clip_bbox.ymax * height;
        clip_bbox.score = score;
        clip_bbox.label = -1;
        boxes.push_back(clip_bbox);
      }
    }
    outs->push_back(boxes);
  }

  auto t4 = Time();
  LOG(INFO) << t2.since_millisecond(t1) << " "
            << t3.since_millisecond(t2) << " "
            << t4.since_millisecond(t3);
}

}  // namespace fd
}  // namespace tron
