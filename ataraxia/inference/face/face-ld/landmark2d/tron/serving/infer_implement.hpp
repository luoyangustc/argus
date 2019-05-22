#ifndef TRON_EXPORT_INFER_IMPLEMENT_HPP
#define TRON_EXPORT_INFER_IMPLEMENT_HPP

#include "common/boxes.hpp"
#include "common/log.hpp"
#include "common/type.hpp"
#include "common/util.hpp"
#include "common/helper.hpp"

#include "core/network.hpp"
#include "proto/tron.pb.h"

#if defined(USE_OpenCV)
#include <opencv2/opencv.hpp>
#endif

namespace Tron {


#define MAX_BATCH_SIZE (16) 

struct TronMatRect{
    cv::Mat im_mat;
    Tron::BoxF face_rect_output;
};

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
#endif

class MobileNetV2{
 public:
  MobileNetV2() = default;
  ~MobileNetV2(){ Release(); }

  void Setup(const tron::MetaNetParam &meta_net_param,
             const VecInt &in_shape,const int gpu_id);

  void Predict(const std::vector<TronMatRect>& im_mat_rect,const VecRectF &rois,
               TronLandmarkOutput*  GLandmarkAspects);

  void Release();

 private:

  void Process(const VecFloat &in_data,
               TronLandmarkOutput* GLandmarkAspects,
              const std::vector<TronMatRect>& im_mat_rect);

  Shadow::Network net_;
  VecFloat in_data_;

  std::string str_aspects,str_landmarks;
  int batch_, in_num_, in_c_, in_h_, in_w_;
  
  bool reshape;
  const float landmark_scale[136]={
   0.22546655,0.42329793,0.22798756,0.49603104,0.23606192,0.5686224
  ,0.25126836,0.64002621,0.2784487,0.7070552,0.32021968,0.76636084
  ,0.37284715,0.81558227,0.43222768,0.85398413,0.50027311,0.86570894
  ,0.56815786,0.85379883,0.62725529,0.81522252,0.6795426,0.76581633
  ,0.72098067,0.70643824,0.74787495,0.63939193,0.76287722,0.56806961
  ,0.77072161,0.49563933,0.77307372,0.42309021,0.28177486,0.37068629
  ,0.31536608,0.34215838,0.36188371,0.33289426,0.41024918,0.33927125
  ,0.45521373,0.35717193,0.54444316,0.35705488,0.58931087,0.33909808
  ,0.63749824,0.33267856,0.68385006,0.34184075,0.71729031,0.37016797
  ,0.49998864,0.41391525,0.50006314,0.46112063,0.50016661,0.50750549
  ,0.50026445,0.55544138,0.44467193,0.589681,0.47106393,0.60015565
  ,0.50019725,0.60760314,0.52928334,0.60012647,0.55552801,0.58964688
  ,0.33543283,0.42512799,0.36442044,0.41155766,0.3978493,0.41110869
  ,0.42781669,0.42765175,0.39688638,0.43601403,0.3634604,0.43638371
  ,0.57175009,0.42757701,0.60170177,0.41099998,0.63507457,0.41143647
  ,0.66393864,0.42497235,0.63610922,0.43622821,0.60276829,0.43589328
  ,0.39883469,0.68130927,0.43570871,0.66366959,0.473645,0.65452444
  ,0.50017537,0.66076641,0.52681312,0.65453653,0.56466597,0.66365442
  ,0.60129465,0.68117379,0.58293278,0.70086329,0.5294545,0.73410688
  ,0.50021722,0.73741709,0.47105424,0.7341431,0.4341824,0.71749142
  ,0.41522898,0.68290855,0.47344854,0.67724316,0.50018261,0.67977186
  ,0.5270066,0.67722852,0.58494966,0.68280836,0.52761093,0.69856826
  ,0.50018237,0.70182311,0.47283021,0.69860521};
};

}  // namespace Tron

#endif  // TRON_EXPORT_INFER_IMPLEMENT_HPP
