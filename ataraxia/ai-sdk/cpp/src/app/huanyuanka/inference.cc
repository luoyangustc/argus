
#include "inference.hpp"

#include "Util.hpp"

namespace tron {
namespace mix {

static Shadow::Logger gLogger;
static Shadow::Profiler gProfiler;

int Inference::Predict(const std::vector<cv::Mat> &requests,
                       std::vector<ResponseMix> *responses) {
  mix_->Predict(requests, responses);

  std::vector<RequestFO> requests_fo;
  for (std::size_t i = 0; i < responses->size(); i++) {
    for (std::size_t j = 0; j < responses->at(i).boxes.size(); j++) {
      requests_fo.emplace_back(requests[i], responses->at(i).boxes[j]);
    }
  }
  std::vector<cv::Mat> responses_fo;
  fo_->Predict(requests_fo, &responses_fo);

  std::vector<cv::Mat> requests_ff;
  for (std::size_t i = 0; i < responses_fo.size(); i++) {
    if (responses_fo[i].empty()) continue;
    requests_ff.push_back(responses_fo[i]);
  }
  std::vector<std::vector<float>> responses_ff;
  ff_->Predict(requests_ff, &responses_ff);

  std::size_t index1 = 0, index2 = 0;
  for (std::size_t i = 0; i < responses->size(); i++) {
    VecBoxF boxes;
    std::vector<std::vector<float>> features;
    for (std::size_t j = 0; j < responses->at(i).boxes.size(); j++, index1++) {
      if (responses_fo[index1].empty()) continue;
      boxes.push_back(responses->at(i).boxes[j]);
      features.push_back(responses_ff[index2++]);
    }
    responses->at(i).boxes = boxes;
    responses->at(i).features = features;
  }
}

}  // namespace mix
}  // namespace tron
