#ifndef TRON_COMMON_IMAGE_HPP
#define TRON_COMMON_IMAGE_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/type.hpp"

namespace tron {

static inline cv::Mat decode_image_buffer(const std::string &im_data) {
  std::vector<char> im_buffer(im_data.begin(), im_data.end());
  try {
    cv::Mat im = cv::imdecode(cv::Mat(im_buffer), 1);
    if (im.channels() == 4) {
      cv::Mat bgr;
      cv::cvtColor(im, bgr, cv::COLOR_BGRA2BGR);
      return bgr;
    } else {
      return im;
    }
  } catch (cv::Exception &e) {
    LOG(WARNING) << e.msg;
    return cv::Mat();
  }
}

static inline void ConvertData(const cv::Mat &im_mat, float *data,
                               const RectF &roi, int channel, int height,
                               int width, int flag = 1,
                               bool transpose = false,
                               const RectF &pad = RectF(0, 0, 0, 0)) {
  CHECK(!im_mat.empty());
  CHECK_NOTNULL(data);

  int c_ = im_mat.channels(), h_ = im_mat.rows, w_ = im_mat.cols;
  int dst_spatial_dim = height * width;

  if (roi.w <= 1 && roi.h <= 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > 1 || roi.y + roi.h > 1) {
      LOG(FATAL) << "crop region overflow! "
                 << roi.x << "," << roi.y << " " << roi.w << " " << roi.h;
      LOG(FATAL) << "Crop region overflow!";
    }
  } else if (roi.w > 1 && roi.h > 1) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > w_ ||
        roi.y + roi.h > h_) {
      LOG(FATAL) << "crop region overflow! "
                 << w_ << "X" << h_ << " "
                 << roi.x << " " << roi.y << " " << roi.w << " " << roi.h;
      LOG(FATAL) << "Crop region overflow!";
    }
  } else {
    LOG(FATAL) << "Crop scale must be the same! " << roi.w << " " << roi.h;
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

  cv::Mat im_resize = cv::Mat(height, width, CV_8UC3);
  if (roi_x != 0 || roi_y != 0 || roi_w != w_ || roi_h != h_) {
    if (pad.w >= 1 && pad.h >= 1) {
      LOG(INFO) << "PAD: "
                << pad.x << " " << pad.y << " " << pad.w << " " << pad.h;
      cv::Mat padded(pad.h, pad.w, CV_8UC3);
      padded.setTo(cv::Scalar::all(0));
      cv::Mat im_tmp = im_mat.clone()(cv_roi);
      im_tmp.copyTo(padded(cv::Rect(pad.x, pad.y, im_tmp.cols, im_tmp.rows)));
      cv::resize(padded, im_resize, cv_size);
    } else {
      cv::Mat im_tmp = im_mat.clone();
      cv::resize(im_tmp(cv_roi), im_resize, cv_size);
    }
  } else {
    if (pad.w >= 1 && pad.h >= 1) {
      LOG(INFO) << "PAD: "
                << pad.x << " " << pad.y << " " << pad.w << " " << pad.h;
      cv::Mat padded(pad.h, pad.w, CV_8UC3);
      padded.setTo(cv::Scalar::all(0));
      im_mat.copyTo(padded(cv::Rect(pad.x, pad.y, im_mat.cols, im_mat.rows)));
      cv::resize(padded, im_resize, cv_size);
    } else {
      cv::resize(im_mat, im_resize, cv_size);
    }
  }

  int dst_h = height, dst_w = width;
  if (transpose) {
    cv::transpose(im_resize.clone(), im_resize);
    dst_h = width, dst_w = height;
  }

  // cv::imwrite("/src/res/tmp/0.jpg", im_resize);

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

}  // namespace tron

#endif  // TRON_COMMON_IMAGE_HPP NOLINT
