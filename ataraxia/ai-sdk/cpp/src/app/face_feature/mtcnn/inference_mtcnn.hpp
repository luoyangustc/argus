#ifndef TRON_FACE_FEATURE_INFERENCE_MTCNN_HPP  // NOLINT
#define TRON_FACE_FEATURE_INFERENCE_MTCNN_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "common/boxes.hpp"
#include "common/image.hpp"
#include "common/type.hpp"
#include "tensord/tensord.hpp"

namespace tron {
namespace ff {

struct BoxInfo {
  BoxF box;
  float box_reg[4], landmark[10];
};

using VecBoxInfo = std::vector<BoxInfo>;

struct Rin {
  Rin() = default;
  Rin(cv::Mat im_mat, BoxF box) : im_mat(im_mat), box(box) {}
  ~Rin() {}
  cv::Mat im_mat;
  BoxF box;
};

class MTCNNRInference {
 public:
  MTCNNRInference() = default;
  ~MTCNNRInference() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape);
  void Predict(const std::vector<Rin> &ins, std::vector<BoxF> *outs);

 private:
  std::shared_ptr<tensord::core::Engine<float>> engine_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
};

struct Oin {
  Oin() = default;
  Oin(cv::Mat im_mat, BoxF box) : im_mat(im_mat), box(box) {}
  ~Oin() {}
  cv::Mat im_mat;
  BoxF box;
};

struct Oout {
  Oout() = default;
  Oout(BoxF box, VecPointF points) : box(box), points(points) {}
  ~Oout() {}
  BoxF box;
  VecPointF points;
};

class MTCNNOInference {
 public:
  MTCNNOInference() = default;
  ~MTCNNOInference() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape);
  void Predict(const std::vector<Oin> &ins, std::vector<Oout> *outs);

 private:
  std::shared_ptr<tensord::core::Engine<float>> engine_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
};

struct Lin {
  Lin() = default;
  Lin(cv::Mat im_mat, BoxF box, VecPointF points)
      : im_mat(im_mat), box(box), points(points) {}
  ~Lin() {}
  cv::Mat im_mat;
  BoxF box;
  VecPointF points;
};

class MTCNNLInference {
 public:
  MTCNNLInference() = default;
  ~MTCNNLInference() {}

  void Setup(std::shared_ptr<tensord::core::Engine<float>> engine,
             const std::vector<int> &in_shape);
  void Predict(const std::vector<Lin> &ins, std::vector<VecPointF> *outs);

 private:
  std::shared_ptr<tensord::core::Engine<float>> engine_;
  VecFloat in_data_;
  int batch_, in_num_, in_c_, in_h_, in_w_;
};

class MTCNNInference {
 public:
  MTCNNInference() = default;
  ~MTCNNInference() {}

  void SetupRNet(std::shared_ptr<MTCNNRInference> rNet) { r_net_ = rNet; }
  void SetupONet(std::shared_ptr<MTCNNOInference> oNet) { o_net_ = oNet; }
  void SetupLNet(std::shared_ptr<MTCNNLInference> lNet) { l_net_ = lNet; }

  void Predict(const std::vector<cv::Mat> &im_mats,
               const VecBoxF &boxes,
               std::vector<VecPointF> *points);

 private:
  std::shared_ptr<MTCNNRInference> r_net_;
  std::shared_ptr<MTCNNOInference> o_net_;
  std::shared_ptr<MTCNNLInference> l_net_;
};

}  // namespace ff
}  // namespace tron

#endif  // TRON_FACE_FEATURE_INFERENCE_MTCNN_HPP NOLINT
