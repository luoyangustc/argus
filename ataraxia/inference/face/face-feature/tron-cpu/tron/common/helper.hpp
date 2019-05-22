#ifndef TRON_COMMON_HELPER_HPP
#define TRON_COMMON_HELPER_HPP

#include "common/boxes.hpp"
#include "common/json.hpp"
#include "common/log.hpp"
#include "common/md5.hpp"
#include "common/type.hpp"
#include "common/util.hpp"

#include "proto/inference.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

enum TronStatus {
  tron_status_success = 200,

  tron_status_method_nullptr = 550,
  tron_status_parse_model_error = 551,
  tron_status_request_data_body_empty = 552,
  tron_status_imdecode_error = 553,
  tron_status_image_size_error = 554,
  tron_status_response_buffer_not_enough = 555,
  tron_status_request_data_attribute_error = 556
};

namespace Tron {

inline const char *get_status_message(int code) {
  switch (code) {
    case 200:
      return "tron_status_success";
    case 550:
      return "tron_status_method_nullptr";
    case 551:
      return "tron_status_parse_model_error";
    case 552:
      return "tron_status_request_data_body_empty";
    case 553:
      return "tron_status_imdecode_error";
    case 554:
      return "tron_status_image_size_error";
    case 555:
      return "tron_status_response_buffer_not_enough";
    case 556:
      return "tron_status_request_data_attribute_error";
    default:
      return "Unknown error";
  }
}

inline bool check_valid_box_pts(const int pts[4][2]){
  //pts: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].
  if(pts[0][0]==pts[3][0] && 
      pts[0][1]==pts[1][1] &&
      pts[1][0]==pts[2][0] &&
      // pts[1][1]==pts[0][1] &&
      // pts[2][0]==pts[1][0] &&
      pts[2][1]==pts[3][1] &&
      // pts[3][0]==pts[0][0] &&
      // pts[3][1]==pts[2][1] &&
      pts[2][0]>pts[0][0] &&
      pts[2][1]>pts[0][1]
  ){
    return true;
  }
  
  return false;
}

inline bool parse_request_boxes(const std::string &attribute, VecBoxF *boxes) {
  char pts_valid_msg[] = ("'attribute' in request data must be a valid json dict string,"
        " and has key 'pts'."
        " pts must be in the form as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."
        " all of x1, x2, y1, y2 can be parsed into int values."
        " And also must have (x2>x1 && y2>y1).");
  const auto &document = Tron::get_document(attribute);

  boxes->clear();

  if (document.HasMember("pts")) {
    const auto &pts = document["pts"];
    int t_pts[4][2];
    
    // CHECK(pts.IsArray());
    try{
      for(int i=0; i<4; i++){
        for(int j=0; j<2; j++){
          t_pts[i][j] = pts[i][j].GetInt();
        }
      }
    }
    catch (...){
      LOG(ERROR) << "Exception when parsing input box pts. " << pts_valid_msg; 
      return false;
    }

    if(!check_valid_box_pts(t_pts))
    {
      LOG(ERROR) << "check_valid_box_pts() failed. Invalid pts for a box. " << pts_valid_msg; 
      return false;
    }

    BoxF box;

    // box.xmin = pts[0][0].GetInt();
    // box.ymin = pts[0][1].GetInt();
    // box.xmax = pts[2][0].GetInt();
    // box.ymax = pts[2][1].GetInt();
    box.xmin = t_pts[0][0];
    box.ymin = t_pts[0][1];
    box.xmax = t_pts[2][0];
    box.ymax = t_pts[2][1];

    boxes->push_back(box);

    return true;
  } else {
    LOG(ERROR) << "Cannot find pts in attribute. " << pts_valid_msg; 
    return false;
  }
}

/**
 * This function is used to parse protobuf message from raw array
 */
inline bool read_proto_from_array(const void *proto_data, int proto_size,
                                  google::protobuf::Message *proto) {
  using google::protobuf::io::ArrayInputStream;
  using google::protobuf::io::CodedInputStream;
  auto *param_array_input = new ArrayInputStream(proto_data, proto_size);
  auto *param_coded_input = new CodedInputStream(param_array_input);
  param_coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
  bool success = proto->ParseFromCodedStream(param_coded_input) &&
                 param_coded_input->ConsumedEntireMessage();
  delete param_coded_input;
  delete param_array_input;
  return success;
}

/**
 * This function is used to parse the serving create params, parse the model
 * buffers and save them to /tmp/models
 */
inline bool parse_create_params(void *params, int params_size,
                                std::string *custom_params,
                                tron::MetaNetParam *meta_net_param_mtcnn,
                                tron::MetaNetParam *meta_net_param_feature) {
  inference::CreateParams create_params;
  bool success = read_proto_from_array(params, params_size, &create_params);
  if (success) {
    *custom_params = create_params.custom_params();
    LOG(INFO) << *custom_params;

    if (create_params.model_files_size() == 2) {
      // const auto &model_file_mtcnn = create_params.model_files(0);
      // const auto &model_file_feature = create_params.model_files(1);

      int j = -1;
      //  std::cout<<"------>\n"<<create_params.model_files(0).name()<<"\n";
      //   std::cout<<"------>\n"<<create_params.model_files(1).name()<<"\n";
      //   std::cout<<"=====>\n"<<create_params.model_files(0).has_name() <<"\n";
      //   std::cout<<"=====>\n"<<create_params.model_files(1).has_name() <<"\n";

      for (int i=0; i<2; i++){
        if (create_params.model_files(i).has_name() 
            && create_params.model_files(i).name().find("mtcnn") != std::string::npos) {
          j = i;
          break;
        }
      }

      if (j < 0)
      {
        LOG(FATAL) << "Cannot find a model file containing string 'mtcnn' !";
        return false;
      }

      const auto &model_file_mtcnn = create_params.model_files(j);
      const auto &model_file_feature = create_params.model_files(1 - j);

      bool parse_success;
      if (model_file_mtcnn.has_body()) {
        const auto *net_data = model_file_mtcnn.body().data();
        auto net_size = model_file_mtcnn.body().size();
        LOG(INFO) << "Model file's body size: " << net_size;
        LOG(INFO) << "Model file's body md5: " << MD5(model_file_mtcnn.body());
        parse_success =
            read_proto_from_array(net_data, net_size, meta_net_param_mtcnn);
      } else {
        LOG(FATAL) << "MTCNN model file has no body!";
        return false;
      }
      if (model_file_feature.has_body()) {
        const auto *net_data = model_file_feature.body().data();
        auto net_size = model_file_feature.body().size();
        LOG(INFO) << "Model file's body size: " << net_size;
        LOG(INFO) << "Model file's body md5: "
                  << MD5(model_file_feature.body());
        parse_success &=
            read_proto_from_array(net_data, net_size, meta_net_param_feature);
      } else {
        LOG(FATAL) << "Feature model file has no body!";
        return false;
      }
      return parse_success;
    } else {
      LOG(FATAL) << "CreateParams has wrong model files!";
      return false;
    }
  } else {
    LOG(FATAL) << "Parsing CreateParams Error!";
    return false;
  }
}
}  // namespace Tron

#endif  // TRON_COMMON_HELPER_HPP
