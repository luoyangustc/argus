#include "forward.hpp"

#include <algorithm>

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "common/boxes.hpp"
#include "common/time.hpp"

namespace tron {
namespace wa {

inline bool compareLabel(const std::pair<int, float> i,
                         const std::pair<int, float> j) {
  return i.second > j.second;
}

void Forward::Setup(const std::vector<std::vector<char>> &net_param_data,
                    const VecInt &in_shape, const int &gpu_id,
                    const ForwardConfig &config) {
  {
    std::vector<std::vector<float>> meanParams =
        {{103.94, 116.78, 123.68, 0.017},
         {103.52, 116.28, 123.675, 1}};
    std::vector<std::vector<char>> params = {
        *const_cast<std::vector<char> *>(&net_param_data[0]),
        *const_cast<std::vector<char> *>(&net_param_data[1])};
    std::vector<VecInt> in_shapes = {{3, 225, 225},
                                     {3, 512, 512}};
    net_ = Shadow::createNet(in_shapes, meanParams, Shadow::bilinear);
    net_->init(gpu_id, params,
               {static_cast<int>(net_param_data[0].size()),
                static_cast<int>(net_param_data[1].size())});
  }

  Base::Setup(in_shape);
  config_ = config;
}

void Forward::Release() { net_->destroy(); }

void Forward::Process(
    std::vector<ForwardRequest>::const_iterator requests_first,
    std::vector<ForwardRequest>::const_iterator requests_last,
    std::vector<ForwardResponse>::iterator responses_first) {
  LOG(INFO) << "forward process begin...";
  auto t1 = Time();
  int ii = 0;
  std::vector<cv::Mat> imgs;
  for (auto cur = requests_first; cur != requests_last; ++cur) {
    auto body = const_cast<ForwardRequest &>(*cur)
                    .mutable_data()
                    ->mutable_body();
    std::vector<float> data(body->size() / sizeof(float));
    memcpy(&data[0], &((*(body))[0]), body->size());
    {
      int h = cur->h(), w = cur->w();
      LOG(INFO) << "H: " << h << ", W: " << w;
      // std::vector<cv::Mat> channels;
      // channels.emplace_back(h, w, CV_32FC1, data.data());
      // channels.emplace_back(h, w, CV_32FC1, data.data() + h * w);
      // channels.emplace_back(h, w, CV_32FC1, data.data() + h * w * 2);
      // cv::Mat img(h, w, CV_32FC3);
      // cv::merge(channels, img);
      cv::Mat img(h, w, CV_8UC3);
      auto b = data.data();
      auto g = data.data() + h * w;
      auto r = data.data() + h * w * 2;
      for (int ih = 0; ih < h; ih++) {
        auto *data_dst = img.ptr<uchar>(ih);
        for (int jw = 0; jw < w; jw++) {
          *data_dst++ = static_cast<uchar>(*b++);
          *data_dst++ = static_cast<uchar>(*g++);
          *data_dst++ = static_cast<uchar>(*r++);
        }
      }
      imgs.push_back(img);
    }
    ii++;
  }
  int size = ii;
  LOG(INFO) << Time().since_millisecond(t1);

  std::vector<std::vector<float>> data1(1);
  {
    std::vector<std::string> layer = {"prob"};
    data1[0].resize(size * 48);
    net_->predict(imgs, layer, data1, 0);
  }

  std::vector<std::vector<float>> data2(1);
  {
    std::vector<std::string> layer = {"detection_out"};
    data2[0].resize(size * 1400);
    Shadow::ShadowStatus status = net_->predict(imgs, layer, data2, 1);
    if (status != Shadow::shadow_status_success) {
      LOG(WARNING) << "det: " << status;
    }
  }

  for (int i = 0; i < size; i++) {
    auto resp = responses_first + i;
    {
      std::vector<std::pair<int, float>> labels;
      for (auto j = 0; j < 48; j++) {
        labels.emplace_back(j, data1[0][48 * i + j]);
      }
      std::sort(labels.begin(), labels.end(), compareLabel);
      for (int j = 0; j < config_.limit; j++) {
        LOG(INFO) << labels[j].first << " " << labels[j].second;
        auto label = resp->add_label();
        label->set_index(labels[j].first);
        label->set_score(labels[j].second);
      }
    }
    for (auto j = i * 200 * 7; j < (i + 1) * 200 * 7; j += 7) {
      // get 7 float value of the a bbox info
      inference::wa::ForwardResponse_Box bbox;
      if (static_cast<int>(data2[0][j + 1]) == -1) {
        break;
      }
      bbox.set_label(data2[0][j + 1]);
      bbox.set_score(data2[0][j + 2]);
      if (bbox.score() < config_.threshold) {
        continue;
      }
      bbox.set_xmin(data2[0][j + 3]);
      bbox.set_ymin(data2[0][j + 4]);
      bbox.set_xmax(data2[0][j + 5]);
      bbox.set_ymax(data2[0][j + 6]);
      // LOG(INFO) << bbox.label() << " " << bbox.score() << " "
      //           << bbox.xmin() << "," << bbox.ymin() << ","
      //           << bbox.xmax() << "," << bbox.ymax();
      if (static_cast<int>(bbox.label()) >= 7 &&
          static_cast<int>(bbox.label()) <= 12) {
        float iw = static_cast<float>(imgs[i].cols);
        float ih = static_cast<float>(imgs[i].rows);
        // image small ,bk_logo not correct
        if (iw < 100 || ih < 100) continue;
        // bk_logo left top or bk_logo right top
        if (bbox.ymax() < 0.333333 &&
            (bbox.xmax() < 0.333333 || bbox.xmin() > 0.666667)) {
          bbox.set_label(7.0);
        } else {
          // bk_logo position not at left top or right top ,so discard it
          continue;
        }
      }
      resp->add_boxes()->CopyFrom(bbox);
    }
  }
  LOG(INFO) << "net forward done.";
}

}  // namespace wa
}  // namespace tron
