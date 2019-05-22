#include "./pre_process.hpp"
#include "./forward.pb.h"
#include "./utils.hpp"

namespace tron {
namespace terror_mixup {

using std::string;
using std::vector;

vector<float> det_pre_process_image(
    cv::Mat_<cv::Vec3b> im_ori,
    const int height,
    const int width) {
  const int channels = 3;
  CHECK_EQ(im_ori.depth(), CV_8U);
  CHECK_EQ(im_ori.channels(), channels);

  cv::Mat_<cv::Vec3b> im;
  cv::resize(im_ori, im, cv::Size(width, height));  // 宽,高

  const float mean_data[] = {103.52, 116.28, 123.675};
  const float pix_scale = 0.017;

  vector<float> input(channels * height * width);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < im.channels(); ++c) {
        const float pix = static_cast<float>(im.at<cv::Vec3b>(h, w)[c]);
        const float pix2 = (pix - mean_data[c]) * pix_scale;
        input.at(c * height * width + h * width + w) = pix2;
      }
    }
  }
  return input;
}

vector<float> cls_pre_process_image(
    cv::Mat_<cv::Vec3b> im_ori,
    int height,
    int width,
    int crop_size) {
  // 这里实现和python不一样，python是先减均值然后裁剪，c++反过来了
  const int channels = 3;
  CHECK_EQ(im_ori.depth(), CV_8U);
  CHECK_EQ(im_ori.channels(), channels);

  cv::Mat_<cv::Vec3f> im;
  {
    cv::Mat_<cv::Vec3f> im_ori2;
    im_ori.convertTo(im_ori2, CV_32FC3);
    cv::resize(im_ori2, im, cv::Size(width, height));  // 宽,高

    const int short_edge = std::min(height, width);
    CHECK_GE(short_edge, crop_size);
    const int yy = (height - crop_size) / 2;
    const int xx = (width - crop_size) / 2;
    im = im(cv::Rect(xx, yy, crop_size, crop_size)).clone();
  }

  height = crop_size;
  width = crop_size;

  const float mean_data[] = {103.94, 116.78, 123.68};
  const float pix_scale = 0.017;

  vector<float> input(channels * height * width);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < im.channels(); ++c) {
        const float pix = static_cast<float>(im.at<cv::Vec3f>(h, w)[c]);
        const float pix2 = (pix - mean_data[c]) * pix_scale;
        input.at(c * height * width + h * width + w) = pix2;
      }
    }
  }
  return input;
}

}  // namespace terror_mixup
}  // namespace tron
