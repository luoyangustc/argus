#ifndef TRON_EXPORT_INFER_IMPLEMENT_HPP
#define TRON_EXPORT_INFER_IMPLEMENT_HPP

#include "common/boxes.hpp"
#include "common/log.hpp"
#include "common/type.hpp"
#include "common/util.hpp"

#include "core/network.hpp"
#include "proto/tron.pb.h"

#if defined(USE_OpenCV)
#include <opencv2/opencv.hpp>
#endif

namespace Tron {

#if defined(USE_OpenCV)
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
  //   std::cout<<"\n\n\nRRRGGGBBB\n\n\n\n";
    data_r = data;
    data_g = data + dst_spatial_dim;
    data_b = data + (dst_spatial_dim << 1);
  } else if (channel == 3 && flag == 1) {
    // Convert to BBBGGGRRR
  //  std::cout<<"\n\n\n BBBGGGRRR\n\n\n\n";
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
#endif

struct BoxInfo {
  BoxF box;
  float box_reg[4], landmark[10];
};

using VecBoxInfo = std::vector<BoxInfo>;

class MTCNN {
 public:
  MTCNN() = default;
  ~MTCNN() { Release(); }

  void Setup(const tron::MetaNetParam &meta_net_param, const VecInt &in_shape,
             int gpu_id);

  void Predict(const cv::Mat &im_mat, const VecBoxF &face_boxes,
               std::vector<VecPointF> *face_points);

  void Release();

 private:
  void Process_net_o(const float *data, const VecInt &in_shape, float threshold,
                     const VecBoxF &net_24_boxes, VecBoxInfo *boxes);

  void BoxRegression(VecBoxInfo &boxes);

  void BoxWithConstrain(VecBoxInfo &boxes, float height, float width);

  Shadow::Network net_o_;
  VecFloat net_o_in_data_, thresholds_;
  VecInt net_o_in_shape_;
  VecBoxInfo net_o_boxes_;
  std::string net_o_conv6_2_, net_o_conv6_3_, net_o_prob1_;
  int net_o_in_c_, net_o_in_h_, net_o_in_w_, net_o_in_num_;
};

class Feature {
 public:
  Feature() = default;
  ~Feature() { Release(); }

  void Setup(const tron::MetaNetParam &meta_net_param, const VecInt &in_shape,
             int gpu_id);

#if defined(USE_OpenCV)
  void Predict(const cv::Mat &im_mat, const VecRectF &rois,
               std::vector<std::map<std::string, VecFloat>> *scores);
#endif

  void Release();

 private:
  void Process(const VecFloat &in_data,
               std::vector<std::map<std::string, VecFloat>> *scores);

  Shadow::Network net_;
  VecFloat in_data_;
  VecInt task_dims_;
  VecString labels_, task_names_;
  std::string prob_str_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
};

}  // namespace Tron

#endif  // TRON_EXPORT_INFER_IMPLEMENT_HPP
