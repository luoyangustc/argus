#include "inference_mtcnn.hpp"

#include <algorithm>

#include "glog/logging.h"

namespace tron {
namespace ff {

inline bool SortBoxesDescend(const BoxInfo &box_a, const BoxInfo &box_b) {
  return box_a.box.score > box_b.box.score;
}

inline float IoU(const BoxF &box_a, const BoxF &box_b, bool is_iom = false) {
  float inter = Boxes::Intersection(box_a, box_b);
  float size_a = Boxes::Size(box_a), size_b = Boxes::Size(box_b);
  if (is_iom) {
    return inter / std::min(size_a, size_b);
  } else {
    return inter / (size_a + size_b - inter);
  }
}

inline VecBoxInfo NMS(const VecBoxInfo &boxes, float threshold,
                      bool is_iom = false) {
  auto all_boxes = boxes;
  std::stable_sort(all_boxes.begin(), all_boxes.end(), SortBoxesDescend);
  for (std::size_t i = 0; i < all_boxes.size(); ++i) {
    auto &box_info_i = all_boxes[i];
    if (box_info_i.box.label == -1) continue;
    for (std::size_t j = i + 1; j < all_boxes.size(); ++j) {
      auto &box_info_j = all_boxes[j];
      if (box_info_j.box.label == -1) continue;
      if (IoU(box_info_i.box, box_info_j.box, is_iom) > threshold) {
        box_info_j.box.label = -1;
        continue;
      }
    }
  }
  VecBoxInfo out_boxes;
  for (const auto &box_info : all_boxes) {
    if (box_info.box.label != -1) {
      out_boxes.push_back(box_info);
    }
  }
  all_boxes.clear();
  return out_boxes;
}

inline void BoxRegression(BoxInfo *box_info) {
  auto &box = box_info->box;
  float box_h = box.ymax - box.ymin + 1, box_w = box.xmax - box.xmin + 1;
  box.xmin += box_info->box_reg[0] * box_w;
  box.ymin += box_info->box_reg[1] * box_h;
  box.xmax += box_info->box_reg[2] * box_w;
  box.ymax += box_info->box_reg[3] * box_h;
}

inline void Box2SquareWithConstrain(BoxInfo *box_info, float height,
                                    float width) {
  auto &box = box_info->box;
  float box_h = box.ymax - box.ymin + 1, box_w = box.xmax - box.xmin + 1;
  float box_l = std::max(box_h, box_w);
  float x_min = box.xmin + (box_w - box_l) / 2;
  float y_min = box.ymin + (box_h - box_l) / 2;
  float x_max = x_min + box_l;
  float y_max = y_min + box_l;
  box.xmin = std::max(0.f, x_min);
  box.ymin = std::max(0.f, y_min);
  box.xmax = std::min(width, x_max);
  box.ymax = std::min(height, y_max);
}

inline void BoxWithConstrain(BoxInfo *box_info, float height,
                             float width) {
  auto &box = box_info->box;
  box.xmin = std::max(0.f, box.xmin);
  box.ymin = std::max(0.f, box.ymin);
  box.xmax = std::min(width, box.xmax);
  box.ymax = std::min(height, box.ymax);
}

////////////////////////////////////////////////////////////////////////////////

void MTCNNRInference::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  engine_ = engine;
}

void MTCNNRInference::Predict(const std::vector<Rin> &ins,
                              std::vector<BoxF> *outs) {
  in_data_.resize(ins.size() * in_num_);

  std::vector<tensord::core::NetIn<float>> requests;

  for (std::size_t i = 0; i < ins.size(); i++) {
    ConvertData(ins[i].im_mat, in_data_.data() + i * in_num_,
                ins[i].box.RectFloat(), in_c_, in_h_, in_w_,
                1, true);
    {
      auto begin = in_data_.begin() + i * in_num_;
      for (auto cur = begin; cur != begin + in_num_; cur++) {
        *cur = (*cur - 127.5) / 128;
      }
    }
    tensord::core::NetIn<float> request;
    request.names.push_back("data");
    request.datas.emplace_back(in_data_.begin() + i * in_num_,
                               in_data_.begin() + i * in_num_ + in_num_);
    requests.push_back(request);
  }

  std::vector<tensord::core::NetOut<float>> responses;
  engine_->Predict(requests, &responses);

  outs->clear();
  for (std::size_t i = 0; i < ins.size(); i++) {
    auto &im_mat = ins[i].im_mat;
    auto &resp = responses[i];
    auto &box = ins[i].box;

    const int h = box.ymax - box.ymin + 1;
    const int w = box.xmax - box.xmin + 1;
    const float crop_h = h <= 1 ? h * im_mat.rows : h;  // ??
    const float crop_w = w <= 1 ? w * im_mat.cols : w;  // ??

    BoxInfo info;
    info.box = box;
    for (int j = 0; j < 4; j++) {
      info.box_reg[j] = resp.datas[0][j];
    }

    BoxRegression(&info);
    Box2SquareWithConstrain(&info, crop_h, crop_w);
    outs->push_back(info.box);
  }
}

void MTCNNOInference::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  engine_ = engine;
}

void MTCNNOInference::Predict(const std::vector<Oin> &ins,
                              std::vector<Oout> *outs) {
  in_data_.resize(ins.size() * in_num_);

  std::vector<tensord::core::NetIn<float>> requests;

  for (std::size_t i = 0; i < ins.size(); i++) {
    ConvertData(ins[i].im_mat, in_data_.data() + i * in_num_,
                ins[i].box.RectFloat(), in_c_, in_h_, in_w_,
                1, true);
    {
      auto begin = in_data_.begin() + i * in_num_;
      for (auto cur = begin; cur != begin + in_num_; cur++) {
        *cur = (*cur - 127.5) / 128;
      }
    }
    tensord::core::NetIn<float> request;
    request.names.push_back("data");
    request.datas.emplace_back(in_data_.begin() + i * in_num_,
                               in_data_.begin() + i * in_num_ + in_num_);
    requests.push_back(request);
  }

  std::vector<tensord::core::NetOut<float>> responses;
  engine_->Predict(requests, &responses);

  outs->clear();
  for (std::size_t i = 0; i < ins.size(); i++) {
    auto &im_mat = ins[i].im_mat;
    auto &resp = responses[i];
    auto &box = ins[i].box;

    const int h = box.ymax - box.ymin + 1, w = box.xmax - box.xmin + 1;
    VecPointF point;
    for (int j = 0; j < 5; j++) {
      point.emplace_back(resp.GetByName("conv6-3")[j] * w + box.xmin - 1,
                         resp.GetByName("conv6-3")[j + 5] * h + box.ymin - 1);
    }
    LOG(INFO)
        << "X1: " << box.xmin << " Y1: " << box.ymin << " "
        << "X2: " << box.xmax << " Y2: " << box.ymax << " "
        << "[[" << point[0].x << "," << point[0].y << "],"
        << "[" << point[1].x << "," << point[1].y << "],"
        << "[" << point[2].x << "," << point[2].y << "],"
        << "[" << point[3].x << "," << point[3].y << "],"
        << "[" << point[4].x << "," << point[4].y << "]]";
    // points->push_back(point);

    BoxInfo info;
    info.box = box;
    for (int j = 0; j < 4; j++) {
      info.box_reg[j] = resp.GetByName("conv6-2")[j];
    }

    LOG(INFO)
        << box.xmin << "," << box.ymin << ","
        << box.xmax << "," << box.ymax << " "
        << info.box_reg[0] << "," << info.box_reg[1] << ","
        << info.box_reg[2] << "," << info.box_reg[3];
    BoxRegression(&info);
    BoxWithConstrain(&info, im_mat.rows, im_mat.cols);

    LOG(INFO)
        << "[[" << info.box.xmin << "," << info.box.ymin << "],"
        << "[" << info.box.xmax << "," << info.box.ymax << "]] "
        << "[[" << box.xmin << "," << box.ymin << "],"
        << "[" << box.xmax << "," << box.ymax << "]]";
    outs->emplace_back(info.box, point);
  }
}

void MTCNNLInference::Setup(
    std::shared_ptr<tensord::core::Engine<float>> engine,
    const std::vector<int> &in_shape) {
  CHECK_EQ(in_shape.size(), 4);
  batch_ = in_shape[0];
  in_c_ = in_shape[1], in_h_ = in_shape[2], in_w_ = in_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

  engine_ = engine;
}

void MTCNNLInference::Predict(const std::vector<Lin> &ins,
                              std::vector<VecPointF> *points) {
  in_data_.resize(ins.size() * in_num_);

  std::vector<tensord::core::NetIn<float>> requests;

  for (std::size_t i = 0; i < ins.size(); i++) {
    auto &im_mat = ins[i].im_mat;
    auto &box = ins[i].box;
    auto &point = ins[i].points;
    // region of each pt
    int patchw = std::floor(
        std::max(box.xmax - box.xmin, box.ymax - box.ymin) * 0.25);
    // make it even
    patchw = ((patchw % 2) == 1) ? (patchw + 1) : patchw;

    int cols = im_mat.cols, rows = im_mat.rows;
    for (int j = 0; j < 5; ++j) {
      auto &p = point[j];
      BoxF part_region(p.x - patchw / 2, p.y - patchw / 2,
                       p.x + patchw / 2, p.y + patchw / 2);
      RectF pad(0, 0, 0, 0);
      if (part_region.xmin < 0) {
        pad.x = 0 - part_region.xmin;
        pad.w = patchw - part_region.xmin;
        pad.h = pad.h >= 1 ? pad.h : patchw;
        part_region.xmin = 0;
      }
      if (part_region.ymin < 0) {
        pad.y = 0 - part_region.ymin;
        pad.h = patchw - part_region.ymin;
        pad.w = pad.w >= 1 ? pad.w : patchw;
        part_region.ymin = 0;
      }
      if (part_region.xmax > cols) {
        pad.w = (pad.w >= 1 ? pad.w : patchw) + part_region.xmax - cols;
        pad.h = pad.h >= 1 ? pad.h : patchw;
        part_region.xmax = cols;
      }
      if (part_region.ymax > rows) {
        pad.h = (pad.h >= 1 ? pad.h : patchw) + part_region.ymax - rows;
        pad.w = pad.w >= 1 ? pad.w : patchw;
        part_region.ymax = rows;
      }

      if (part_region.xmin > cols || part_region.ymin > rows ||
          part_region.xmax < 0 || part_region.ymax < 0 ||
          part_region.xmax - part_region.xmin < 1 ||
          part_region.ymax - part_region.ymin < 1) {
        LOG(INFO) << "?? "
                  << p.x << " " << p.y << " "
                  << box.xmin << " "
                  << box.ymin << " "
                  << box.xmax << " "
                  << box.ymax << " ";

        auto begin = in_data_.data() + i * in_num_ + j * in_num_ / 5;
        for (auto cur = begin; cur != begin + in_num_ / 5; cur++) {
          *cur = 0;
        }
      } else {
        ConvertData(im_mat,
                    in_data_.data() + i * in_num_ + j * in_num_ / 5,
                    part_region.RectFloat(), in_c_ / 5, in_h_, in_w_,
                    1, true, pad);
      }
      {
        auto begin = in_data_.data() + i * in_num_ + j * in_num_ / 5;
        for (auto cur = begin; cur != begin + in_num_ / 5; cur++) {
          *cur = (*cur - 127.5) / 128;
        }
      }
    }
    tensord::core::NetIn<float> request;
    request.names.push_back("data");
    request.datas.emplace_back(in_data_.begin() + i * in_num_,
                               in_data_.begin() + i * in_num_ + in_num_);

    requests.push_back(request);
  }

  std::vector<tensord::core::NetOut<float>> responses;
  engine_->Predict(requests, &responses);

  points->clear();
  for (std::size_t i = 0; i < ins.size(); i++) {
    auto resp = responses[i];
    auto &box = ins[i].box;
    auto &point = ins[i].points;
    // region of each pt
    int patchw = std::floor(
        std::max(box.xmax - box.xmin, box.ymax - box.ymin) * 0.25);
    // make it even
    patchw = ((patchw % 2) == 1) ? (patchw + 1) : patchw;

    VecPointF _points(5);
    for (int j = 0; j < 5; j++) {
      auto landmark = resp.GetByName("fc5_" + std::to_string(j + 1));
      if (std::abs(landmark[0] - 0.5) > 0.35) {
        landmark[0] = 0.5;
      }
      if (std::abs(landmark[1] - 0.5) > 0.35) {
        landmark[1] = 0.5;
      }
      _points[j].x = point[j].x - 0.5 * patchw + landmark[0] * patchw;
      _points[j].y = point[j].y - 0.5 * patchw + landmark[1] * patchw;
    }
    LOG(INFO)
        << "[[" << _points[0].x << "," << _points[0].y << "],"
        << "[" << _points[1].x << "," << _points[1].y << "],"
        << "[" << _points[2].x << "," << _points[2].y << "],"
        << "[" << _points[3].x << "," << _points[3].y << "],"
        << "[" << _points[4].x << "," << _points[4].y << "]]";
    points->push_back(_points);
  }
}

void MTCNNInference::Predict(const std::vector<cv::Mat> &im_mats,
                             const VecBoxF &boxes,
                             std::vector<VecPointF> *points) {
  // std::vector<cv::Mat> _im_mats;
  VecBoxF _boxes = boxes;
  // // BGR->RGB
  // for (std::size_t i = 0; i < im_mats.size(); i++) {
  //   assert(!im_mats[i].empty());
  //   cv::Mat tmp = im_mats[i].clone();
  //   // cv::imwrite("/src/res/tmp/0.jpg", tmp);
  //   _im_mats.push_back(tmp);
  // }

  if (false && r_net_.use_count() > 0) {
    std::vector<Rin> ins;
    for (std::size_t i = 0; i < im_mats.size(); i++) {
      ins.emplace_back(im_mats[i], _boxes[i]);
    }
    std::vector<BoxF> outs;
    r_net_->Predict(ins, &outs);
    _boxes = outs;
  }

  std::vector<Oin> o_ins;
  for (std::size_t i = 0; i < im_mats.size(); i++) {
    o_ins.emplace_back(im_mats[i], _boxes[i]);
  }
  std::vector<Oout> o_outs;
  o_net_->Predict(o_ins, &o_outs);
  points->clear();
  for (std::size_t i = 0; i < o_outs.size(); i++) {
    points->push_back(o_outs[i].points);
  }

  if (l_net_.use_count() > 0) {
    std::vector<Lin> ins;
    for (std::size_t i = 0; i < im_mats.size(); i++) {
      ins.emplace_back(im_mats[i], o_outs[i].box, o_outs[i].points);
    }
    std::vector<VecPointF> outs;
    l_net_->Predict(ins, &outs);
    points->clear();
    for (std::size_t i = 0; i < outs.size(); i++) {
      points->push_back(outs[i]);
    }
  }
}

}  // namespace ff
}  // namespace tron
