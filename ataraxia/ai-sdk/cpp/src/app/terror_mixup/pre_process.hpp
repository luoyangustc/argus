#ifndef TRON_TERROR_MIXUP_PRE_PROCESS
#define TRON_TERROR_MIXUP_PRE_PROCESS

#include <opencv2/opencv.hpp>
#include <string>
#include <utils.hpp>
#include <vector>
#include "./forward.pb.h"
#include "./utils.hpp"

namespace tron {
namespace terror_mixup {

using std::string;
using std::vector;

vector<float> det_pre_process_image(
    cv::Mat_<cv::Vec3b> im_ori,
    int height = 320,
    int width = 320);
vector<float> cls_pre_process_image(
    cv::Mat_<cv::Vec3b> im_ori,
    int height = 256,
    int width = 256,
    int crop_size = 225);

}  // namespace terror_mixup
}  // namespace tron
#endif  // TRON_FACE_FEATURE_FORWARD_FEATURE_HPP NOLINT
