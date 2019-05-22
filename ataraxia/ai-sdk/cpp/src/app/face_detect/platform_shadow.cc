
#include "platform_shadow.hpp"

#include <string>

#include "glog/logging.h"

#include "common/boxes.hpp"
#include "common/time.hpp"
#include "proto/tron.pb.h"

namespace tron {
namespace fd {

std::shared_ptr<tensord::core::Net<float>> Shadow::Create(
    tensord::proto::Model model,
    tensord::proto::Instance::Kind kind,
    int gpu_id,
    int batch_size) {
  auto net = std::make_shared<Shadow>();
  net->model_ = model;
  net->kind_ = kind;
  net->gpu_id_ = gpu_id;
  net->batch_size_ = batch_size;
  return net;
}

void Shadow::Init() {
  const char *model_params;
  int model_params_size;
  for (int i = 0; i < model_.file_size(); i++) {
    if (model_.mutable_file(i)->name() == "net.tronmodel") {
      model_params = model_.mutable_file(i)->mutable_body()->data();
      model_params_size = model_.mutable_file(i)->mutable_body()->size();
    }
  }
  CHECK_GT(model_params_size, 0);
  tron::MetaNetParam meta_net_param;
  auto success = meta_net_param.ParseFromArray(model_params, model_params_size);
  CHECK_EQ(success, true);

  net_.Setup(meta_net_param.network(0), gpu_id_);

  std::map<std::string, VecInt> shape_map;
  for (int i = 0; i < model_.input_size(); i++) {
    std::vector<int> shapes = {batch_size_};
    for (int j = 0; j < model_.mutable_input(i)->shape_size(); j++) {
      shapes.push_back(model_.mutable_input(i)->shape(j));
    }
    shape_map[model_.mutable_input(i)->name()] = shapes;
  }
  net_.Reshape(shape_map);

  {
    std::vector<std::string> input_keys;
    std::vector<int> input_shapes;
    for (int i = 0; i < model_.input_size(); i++) {
      input_keys.push_back(model_.input(i).name());
      int shape = 1;
      for (int j = 0; j < model_.input(i).shape_size(); j++) {
        shape *= model_.input(i).shape(j);
      }
      input_shapes.push_back(shape);
      input_shape_[model_.input(i).name()] = shape;
    }
    input_buf_ = Buf<float>(batch_size_, input_keys, input_shapes);
  }
  {
    std::vector<std::string> output_keys;
    std::vector<int> output_shapes;
    for (int i = 0; i < model_.output_size(); i++) {
      output_keys.push_back(model_.output(i).name());
      int shape = 1;
      for (int j = 0; j < model_.output(i).shape_size(); j++) {
        shape *= model_.output(i).shape(j);
      }
      output_shapes.push_back(shape);
      output_shape_[model_.output(i).name()] = shape;
    }
    output_buf_ = Buf<float>(batch_size_, output_keys, output_shapes);
  }
}

void Shadow::Predict(const std::vector<tensord::core::NetIn<float>> &ins,
                     std::vector<tensord::core::NetOut<float>> *outs) {
  for (std::size_t i = 0; i < ins.size(); i++) {
    for (std::size_t j = 0; j < ins[i].names.size(); j++) {
      input_buf_.Copy(ins[i].names[j], i, ins[i].datas[j].data());
    }
  }
  std::map<std::string, float *> data_map;
  for (const std::string &name : ins[0].names) {
    data_map[name] = input_buf_.Get(name, 0);
  }
  net_.Forward(data_map);
  PostForward(ins, outs);
}

void Shadow::PostForward(const std::vector<tensord::core::NetIn<float>> &ins,
                         std::vector<tensord::core::NetOut<float>> *outs) {
  for (int i = 0; i < model_.output_size(); i++) {
    memcpy(output_buf_.Get(model_.output(i).name(), 0),
           net_.GetBlobDataByName(model_.output(i).name()),
           output_shape_[model_.output(i).name()] * batch_size_);
  }
  for (std::size_t i = 0; i < ins.size(); i++) {
    tensord::core::NetOut<float> out;
    for (auto iter = output_shape_.begin();
         iter != output_shape_.end();
         iter++) {
      out.names.push_back(iter->first);
      std::vector<float> _buf(iter->second);
      memcpy(_buf.data(),
             output_buf_.Get(iter->first, i),
             iter->second * sizeof(float));
      out.datas.emplace_back(_buf);
    }
    outs->push_back(out);
  }
}

void Shadow::Release() { net_.Release(); }

////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<tensord::core::Net<float>> ShadowFD::Create(
    tensord::proto::Model model,
    tensord::proto::Instance::Kind kind,
    int gpu_id,
    int batch_size) {
  auto net = std::make_shared<ShadowFD>();
  net->model_ = model;
  net->kind_ = kind;
  net->gpu_id_ = gpu_id;
  net->batch_size_ = batch_size;
  return net;
}

void ShadowFD::Init() {
  Shadow::Init();

  output_buf_.batch_size_ = batch_size_;

  int size = 0;
  output_buf_.keys_["odm_loc"] = std::pair<int, int>(size, 65472);
  output_shape_["odm_loc"] = 65472;
  size += batch_size_ * 65472;
  output_buf_.keys_["odm_conf_flatten"] = std::pair<int, int>(size, 32736);
  output_shape_["odm_conf_flatten"] = 32736;
  size += batch_size_ * 32736;
  output_buf_.keys_["arm_priorbox"] = std::pair<int, int>(size, 1 * 2 * 65472);
  output_shape_["arm_priorbox"] = 1 * 2 * 65472;
  size += 1 * 2 * 65472;
  output_buf_.keys_["arm_conf_flatten"] = std::pair<int, int>(size, 32736);
  output_shape_["arm_conf_flatten"] = 32736;
  size += batch_size_ * 32736;
  output_buf_.keys_["arm_loc"] = std::pair<int, int>(size, 65472);
  output_shape_["arm_loc"] = 65472;
  size += batch_size_ * 65472;

  output_buf_.buf_.resize(size);
}

void ShadowFD::PostForward(const std::vector<tensord::core::NetIn<float>> &ins,
                           std::vector<tensord::core::NetOut<float>> *outs) {
  memcpy(output_buf_.Get("odm_loc", 0),
         net_.GetBlobDataByName("odm_loc"),
         output_shape_["odm_loc"] * batch_size_ * sizeof(float));
  memcpy(output_buf_.Get("odm_conf_flatten", 0),
         net_.GetBlobDataByName("odm_conf_flatten"),
         output_shape_["odm_conf_flatten"] * batch_size_ * sizeof(float));
  memcpy(output_buf_.Get("arm_priorbox", 0),
         net_.GetBlobDataByName("arm_priorbox"),
         output_shape_["arm_priorbox"] * sizeof(float));
  memcpy(output_buf_.Get("arm_conf_flatten", 0),
         net_.GetBlobDataByName("arm_conf_flatten"),
         output_shape_["arm_conf_flatten"] * batch_size_ * sizeof(float));
  memcpy(output_buf_.Get("arm_loc", 0),
         net_.GetBlobDataByName("arm_loc"),
         output_shape_["arm_loc"] * batch_size_ * sizeof(float));

  auto n = 65472 / 4;
  auto n4 = n * 4;
  for (std::size_t i = 0; i < ins.size(); i++) {
    auto ol = output_buf_.Get("odm_loc", i);
    auto ap = output_buf_.Get("arm_priorbox", 0);
    auto al = output_buf_.Get("arm_loc", i);

    auto oc = output_buf_.Get("odm_conf_flatten", i);
    auto ac = output_buf_.Get("arm_conf_flatten", i);

    for (int j = 0; j < n; j++) {
      int x1 = j * 4 + 0;
      int y1 = j * 4 + 1;
      int x2 = j * 4 + 2;
      int y2 = j * 4 + 3;

      float xmin = *(ap + x1);
      float ymin = *(ap + y1);
      float xmax = *(ap + x2);
      float ymax = *(ap + y2);

      float width = xmax - xmin;
      float height = ymax - ymin;
      float center_x = (xmin + xmax) / 2.f;
      float center_y = (ymin + ymax) / 2.f;

      float bbox_center_x = *(ap + n4 + x1) * *(al + x1) * width + center_x;
      float bbox_center_y = *(ap + n4 + y1) * *(al + y1) * height + center_y;
      float bbox_width = std::exp(*(ap + n4 + x2) * *(al + x2)) * width;
      float bbox_height = std::exp(*(ap + n4 + y2) * *(al + y2)) * height;

      xmin = bbox_center_x - bbox_width / 2.f;
      ymin = bbox_center_y - bbox_height / 2.f;
      xmax = bbox_center_x + bbox_width / 2.f;
      ymax = bbox_center_y + bbox_height / 2.f;

      width = xmax - xmin;
      height = ymax - ymin;
      center_x = (xmin + xmax) / 2.f;
      center_y = (ymin + ymax) / 2.f;

      bbox_center_x = *(ap + n4 + x1) * *(ol + x1) * width + center_x;
      bbox_center_y = *(ap + n4 + y1) * *(ol + y1) * height + center_y;
      bbox_width = std::exp(*(ap + n4 + x2) * *(ol + x2)) * width;
      bbox_height = std::exp(*(ap + n4 + y2) * *(ol + y2)) * height;

      *(ol + x1) = bbox_center_x - bbox_width / 2.f;
      *(ol + y1) = bbox_center_y - bbox_height / 2.f;
      *(ol + x2) = bbox_center_x + bbox_width / 2.f;
      *(ol + y2) = bbox_center_y + bbox_height / 2.f;

      if (*(ac + 2 * j + 1) < 0.01) {
        *(oc + 2 * j + 0) = 1.0;
        *(oc + 2 * j + 1) = 0.0;
      }
    }
  }

  for (std::size_t i = 0; i < ins.size(); i++) {
    tensord::core::NetOut<float> out;
    {
      out.names.push_back("odm_loc");
      std::vector<float> _buf(output_shape_["odm_loc"]);
      memcpy(_buf.data(),
             output_buf_.Get("odm_loc", i),
             _buf.size() * sizeof(float));
      out.datas.emplace_back(_buf);
    }
    {
      out.names.push_back("odm_conf_flatten");
      std::vector<float> _buf(output_shape_["odm_conf_flatten"]);
      memcpy(_buf.data(),
             output_buf_.Get("odm_conf_flatten", i),
             _buf.size() * sizeof(float));
      out.datas.emplace_back(_buf);
    }
    outs->push_back(out);
  }
}

}  // namespace fd
}  // namespace tron
