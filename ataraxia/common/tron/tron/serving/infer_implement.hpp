#ifndef TRON_EXPORT_INFER_IMPLEMENT_HPP
#define TRON_EXPORT_INFER_IMPLEMENT_HPP

#include "common/boxes.hpp"
#include "common/log.hpp"
#include "common/type.hpp"
#include "common/util.hpp"

#include "core/network.hpp"
#include "proto/tron.pb.h"

#include <opencv2/opencv.hpp>

namespace Tron {

static inline void ConvertData(const cv::Mat &im_mat, float *data,
                               const RectF &roi, int channel, int height,
                               int width, int flag = 1,
                               bool transpose = false) {
  CHECK(!im_mat.empty());
  CHECK_NOTNULL(data);

  int c_ = im_mat.channels(), h_ = im_mat.rows, w_ = im_mat.cols;
  int dst_spatial_dim = height * width;

  if (roi.w <= 1 && roi.h <= 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > 1 || roi.y + roi.h > 1) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else if (roi.w > 1 && roi.h > 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > w_ || roi.y + roi.h > h_) {
      LOG(FATAL) << "Crop region overflow!";
    }
  } else {
    LOG(FATAL) << "Crop scale must be the same!";
  }

  float *data_r = nullptr, *data_g = nullptr, *data_b = nullptr,
        *data_gray = nullptr;
  if (channel == 3 && flag == 0) {
    // Convert to RRRGGGBBB
    data_r = data;
    data_g = data + dst_spatial_dim;
    data_b = data + (dst_spatial_dim << 1);
  } else if (channel == 3 && flag == 1) {
    // Convert to BBBGGGRRR
    data_r = data + (dst_spatial_dim << 1);
    data_g = data + dst_spatial_dim;
    data_b = data;
  } else if (channel == 1) {
    // Convert to Gray
    data_gray = data;
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }

  auto roi_x = static_cast<int>(roi.w <= 1 ? roi.x * w_ : roi.x);
  auto roi_y = static_cast<int>(roi.h <= 1 ? roi.y * h_ : roi.y);
  auto roi_w = static_cast<int>(roi.w <= 1 ? roi.w * w_ : roi.w);
  auto roi_h = static_cast<int>(roi.h <= 1 ? roi.h * h_ : roi.h);

  cv::Rect cv_roi(roi_x, roi_y, roi_w, roi_h);
  cv::Size cv_size(width, height);

  cv::Mat im_resize;
  if (roi_x != 0 || roi_y != 0 || roi_w != w_ || roi_h != h_) {
    cv::resize(im_mat(cv_roi), im_resize, cv_size);
  } else {
    cv::resize(im_mat, im_resize, cv_size);
  }

  int dst_h = height, dst_w = width;
  if (transpose) {
    cv::transpose(im_resize, im_resize);
    dst_h = width, dst_w = height;
  }

  if (channel == 3) {
    CHECK_EQ(c_, 3);
    for (int h = 0; h < dst_h; ++h) {
      const auto *data_src = im_resize.ptr<uchar>(h);
      for (int w = 0; w < dst_w; ++w) {
        *data_b++ = static_cast<float>(*data_src++);
        *data_g++ = static_cast<float>(*data_src++);
        *data_r++ = static_cast<float>(*data_src++);
      }
    }
  } else if (channel == 1) {
    cv::Mat im_gray;
    cv::cvtColor(im_resize, im_gray, cv::COLOR_BGR2GRAY);
    for (int h = 0; h < dst_h; ++h) {
      const auto *data_src = im_gray.ptr<uchar>(h);
      for (int w = 0; w < dst_w; ++w) {
        *data_gray++ = static_cast<float>(*data_src++);
      }
    }
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }
}

class Classification {
 public:
  Classification() = default;

  void Setup(const tron::MetaNetParam &meta_net_param, int gpu_id);

  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<std::map<std::string, VecFloat>> *scores);

 private:
  void Process(const VecFloat &in_data,
               std::vector<std::map<std::string, VecFloat>> *scores);

  Shadow::Network net_;
  VecFloat in_data_;
  std::string in_str_, prob_str_;
  int num_classes_, batch_, in_num_, in_c_, in_h_, in_w_;
};

}  // namespace Tron

#endif  // TRON_EXPORT_INFER_IMPLEMENT_HPP
