
#include "glog/logging.h"

#include "common/archiver.hpp"
#include "common/image.hpp"
#include "common/time.hpp"
#include "infer.hpp"
#include "proto/inference.pb.h"

int main(int, char const *[]) {
  std::vector<float> data(255 * 255 * 3);
  cv::Mat mat = cv::imread("/Users/guaguasong/88.jpg");

  tron::ConvertData(
      mat, &data[0],
      {0, 0, static_cast<float>(mat.cols), static_cast<float>(mat.rows)}, 3,
      255, 255);

  cv::Mat outputMat(255, 255, CV_8UC3);
  cv::Mat channelB(255, 255, CV_16SC1, data[0]);
  cv::Mat channelG(255, 255, CV_16SC1, data[255 * 255]);
  cv::Mat channelR(255, 255, CV_16SC1, data[255 * 255 * 2]);
  std::vector<cv::Mat> channels{channelB, channelG, channelR};
  cv::merge(channels, outputMat);

  cv::imwrite("/Users/guaguasong/99.jpg", outputMat);

  return 0;
}
