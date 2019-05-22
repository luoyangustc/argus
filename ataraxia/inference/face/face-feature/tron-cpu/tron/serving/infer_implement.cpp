#include "infer_implement.hpp"

namespace Tron {

inline bool SortBoxesDescend(const BoxInfo &box_a, const BoxInfo &box_b) {
  return box_a.box.score > box_b.box.score;
}

inline float IoU(const BoxF &box_a, const BoxF &box_b, bool is_iom = false) {
  float inter = Boxes::Intersection(box_a, box_b);
  float size_a = Boxes::Size(box_a), size_b = Boxes::Size(box_b);
  if (is_iom) {
    return inter / std::min(size_a, size_b);
  } else {
    return inter / (size_a + size_b - inter);
  }
}

inline VecBoxInfo NMS(const VecBoxInfo &boxes, float threshold,
                      bool is_iom = false) {
  auto all_boxes = boxes;
  std::stable_sort(all_boxes.begin(), all_boxes.end(), SortBoxesDescend);
  for (int i = 0; i < all_boxes.size(); ++i) {
    auto &box_info_i = all_boxes[i];
    if (box_info_i.box.label == -1) continue;
    for (int j = i + 1; j < all_boxes.size(); ++j) {
      auto &box_info_j = all_boxes[j];
      if (box_info_j.box.label == -1) continue;
      if (IoU(box_info_i.box, box_info_j.box, is_iom) > threshold) {
        box_info_j.box.label = -1;
        continue;
      }
    }
  }
  VecBoxInfo out_boxes;
  for (const auto &box_info : all_boxes) {
    if (box_info.box.label != -1) {
      out_boxes.push_back(box_info);
    }
  }
  all_boxes.clear();
  return out_boxes;
}

void MTCNN::Setup(const tron::MetaNetParam &meta_net_param,
                  const VecInt &in_shape, int gpu_id) {
  net_o_.Setup(gpu_id);

  net_o_.LoadModel(meta_net_param.network(2));

  net_o_conv6_2_ = net_o_.out_blob()[0];
  net_o_conv6_3_ = net_o_.out_blob()[1];
  net_o_prob1_ = net_o_.out_blob()[2];

  net_o_in_shape_ = net_o_.GetBlobByName<float>("data")->shape();
  net_o_in_c_ = net_o_in_shape_[1];
  net_o_in_h_ = net_o_in_shape_[2];
  net_o_in_w_ = net_o_in_shape_[3];
  net_o_in_num_ = net_o_in_c_ * net_o_in_h_ * net_o_in_w_;

  thresholds_ = {0.6f, 0.6f, 0.7f};
}

void MTCNN::Predict(const cv::Mat &im_mat, const VecBoxF &face_boxes,
                    std::vector<VecPointF> *face_points) {
  net_o_boxes_.clear();

  const auto &net_r_boxes_ = face_boxes;
  if (net_r_boxes_.empty()) return;

  net_o_in_shape_[0] = static_cast<int>(net_r_boxes_.size());
  net_o_in_data_.resize(net_o_in_shape_[0] * net_o_in_num_);
  for (int n = 0; n < net_r_boxes_.size(); ++n) {
    const auto &net_24_box = net_r_boxes_[n];
    ConvertData(im_mat, net_o_in_data_.data() + n * net_o_in_num_,
                net_24_box.RectFloat(), net_o_in_c_, net_o_in_h_, net_o_in_w_,
                1, true);
  }

  Process_net_o(net_o_in_data_.data(), net_o_in_shape_, thresholds_[2],
                net_r_boxes_, &net_o_boxes_);
  // BoxRegression(net_o_boxes_);
  // net_o_boxes_ = NMS(net_o_boxes_, 0.7, true);
  // BoxWithConstrain(net_o_boxes_, im_mat.rows, im_mat.cols);

  VecBoxF boxes;
  face_points->clear();
  for (const auto &box_info : net_o_boxes_) {
    boxes.push_back(box_info.box);
    VecPointF mark_points;
 //  std::cout<<"landmark....\n";
    for (int k = 0; k < 5; ++k) {
      mark_points.emplace_back(box_info.landmark[2 * k],
                               box_info.landmark[2 * k + 1]);
 //  std::cout<<box_info.landmark[2 * k]<<","<<box_info.landmark[2 * k+1]<<".\n";
    }
    face_points->push_back(mark_points);
  }
}

void MTCNN::Release() { net_o_.Release(); }

void MTCNN::Process_net_o(const float *data, const VecInt &in_shape,
                          float threshold,   const VecBoxF &net_24_boxes,
                          VecBoxInfo *boxes) {
  std::map<std::string, VecInt> shape_map;
  std::map<std::string, float *> data_map;
  shape_map["data"] = in_shape;
  data_map["data"] = const_cast<float *>(data);

  net_o_.Reshape(shape_map);
  net_o_.Forward(data_map);

  const auto *loc_data = net_o_.GetBlobDataByName<float>(net_o_conv6_2_);
  const auto *mark_data = net_o_.GetBlobDataByName<float>(net_o_conv6_3_);
  const auto *conf_data = net_o_.GetBlobDataByName<float>(net_o_prob1_);

  boxes->clear();
  for (int b = 0; b < in_shape[0]; ++b) {
    int loc_offset = 4 * b, mark_offset = 10 * b;
    float conf = conf_data[2 * b + 1];
    // if (conf > threshold) {
    BoxInfo box_info;
    box_info.box = net_24_boxes[b];
    box_info.box.score = conf;
    box_info.box.label = 1;
    for (int k = 0; k < 4; ++k) {
      box_info.box_reg[k] = loc_data[loc_offset + k];
    }
    float box_h = box_info.box.ymax - box_info.box.ymin + 1,
          box_w = box_info.box.xmax - box_info.box.xmin + 1;
      //     std::cout<<"box_h="<<box_h<<",box_w="<<box_w<<"\n";
    for (int k = 0; k < 5; ++k) {
      box_info.landmark[2 * k] =
          mark_data[mark_offset + k] * box_w + box_info.box.xmin;
      box_info.landmark[2 * k + 1] =
          mark_data[mark_offset + k + 5] * box_h + box_info.box.ymin;
    }
    boxes->push_back(box_info);
    // }
  }
}

void MTCNN::BoxRegression(VecBoxInfo &boxes) {
  for (auto &box_info : boxes) {
    auto &box = box_info.box;
    float box_h = box.ymax - box.ymin + 1, box_w = box.xmax - box.xmin + 1;
    box.xmin += box_info.box_reg[0] * box_w;
    box.ymin += box_info.box_reg[1] * box_h;
    box.xmax += box_info.box_reg[2] * box_w;
    box.ymax += box_info.box_reg[3] * box_h;
  }
}

void MTCNN::BoxWithConstrain(VecBoxInfo &boxes, float height, float width) {
  for (auto &box_info : boxes) {
    auto &box = box_info.box;
    box.xmin = std::max(0.f, box.xmin);
    box.ymin = std::max(0.f, box.ymin);
    box.xmax = std::min(width, box.xmax);
    box.ymax = std::min(height, box.ymax);
  }
}

void Feature::Setup(const tron::MetaNetParam &meta_net_param,
                    const VecInt &in_shape, int gpu_id) {

  net_.LoadModel(meta_net_param.network(0));

  auto data_shape = net_.GetBlobByName<float>("data")->shape();
  CHECK_EQ(data_shape.size(), 4);
  CHECK_EQ(in_shape.size(), 1);
  if (data_shape[0] != in_shape[0]) {
    data_shape[0] = in_shape[0];
    std::map<std::string, VecInt> shape_map;
    shape_map["data"] = data_shape;
    net_.Reshape(shape_map);
  }

  const auto &out_blob = net_.out_blob();
 // std::cout<<"out_blob.size()="<<out_blob.size()<<"\n";
  CHECK_EQ(out_blob.size(), 1);
  prob_str_ = out_blob[0];

  batch_ = data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;

  in_data_.resize(batch_ * in_num_);
  // std::cout<<"batch_="<<batch_<<",c="<<in_c_<<",h="<<in_h_<<",w="<<in_w_<<"\n";

  task_names_ = VecString{"score"};
  
  task_dims_ = net_.num_class();
 // std::cout<<"task_dims_=" << task_dims_.size()<<"\n";
  CHECK_EQ(task_names_.size(), task_dims_.size());
  int num_dim = 0;
  for (const auto dim : task_dims_) {
    num_dim += dim;
  }
  CHECK_EQ(num_dim, net_.GetBlobByName<float>(prob_str_)->num());
  labels_ = net_.get_repeated_argument<std::string>("labels", VecString{});

}

void Feature::Predict(const cv::Mat &im_mat, const VecRectF &rois,
                      std::vector<std::map<std::string, VecFloat>> *scores) {
  cv::cvtColor(im_mat, im_mat, CV_BGR2RGB);      //added     
  CHECK_EQ(512, net_.GetBlobByName<float>("fc1")->num());          
  CHECK_LE(rois.size(), batch_);
  for (int b = 0; b < rois.size(); ++b) {
    ConvertData(im_mat, in_data_.data() + b * in_num_, rois[b], in_c_, in_h_,
                in_w_);
  }

  Process(in_data_, scores);

  CHECK_EQ(scores->size(), rois.size());
}

void Feature::Release() { net_.Release(); }

void Feature::Process(const VecFloat &in_data,
                      std::vector<std::map<std::string, VecFloat>> *scores) {
  std::map<std::string, float *> data_map;
  data_map["data"] = const_cast<float *>(in_data.data());

  net_.Forward(data_map);

  const auto *softmax_data = net_.GetBlobDataByName<float>(prob_str_);
 //std::cout<<"prob_str_="<<prob_str_<<"\n";
  scores->clear();
  int offset = 0;
  for (int b = 0; b < batch_; ++b) {
    std::map<std::string, VecFloat> score_map;
    for (int n = 0; n < task_dims_.size(); ++n) {
      const auto &name = task_names_[n];
      int dim = task_dims_[n];
  //     std::cout<<"dim="<<dim<<"\n";
      VecFloat temp(softmax_data + offset, softmax_data + offset + dim);
            // norm feature
    float sum=0.;
    for(int i=0;i<dim;++i){
      sum+=temp[i]*temp[i];
    }

     sum=std::sqrt(sum);
     for(int i=0;i<dim;++i){
     temp[i]/=sum;
     }
      score_map[name] = temp;
      offset += dim;
    }
    scores->push_back(score_map);
  }
}

}  // namespace Tron
