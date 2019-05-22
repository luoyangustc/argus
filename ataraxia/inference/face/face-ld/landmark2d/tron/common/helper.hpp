#ifndef TRON_COMMON_HELPER_HPP
#define TRON_COMMON_HELPER_HPP

#include "common/boxes.hpp"
#include "common/json.hpp"
#include "common/log.hpp"
#include "common/md5.hpp"
#include "common/type.hpp"
#include "common/util.hpp"
#include "proto/inference.pb.h"
#include "proto/tron.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <opencv2/opencv.hpp>
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

struct TronLandmarkAspectsPara{
  const static int aspects_num=3;
  const static int landmarks_x_y_num=136;
  float aspects[aspects_num];
  float landmark[landmarks_x_y_num];
};

struct TronLandmarkOutput {
  std::vector<TronLandmarkAspectsPara> objects;
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
      return " tron_status_request_data_attribute_error";
    default:
      return "Unknown error";
  }
}

inline bool check_valid_box_pts(const float pts[4][2]){

  if(pts[0][0]==pts[3][0] &&
      pts[0][1]==pts[1][1] &&
      pts[1][0]==pts[2][0] &&
      pts[2][1]==pts[3][1] &&
      pts[2][0]>pts[0][0] &&
      pts[2][1]>pts[0][1]
     
  ){
    return true;
  }

  return false;
}


inline bool parse_request_boxes(const std::string &attribute, VecBoxF *boxes,const cv::Size imsize) {
  const auto &document = Tron::get_document(attribute);
  char pts_valid_msg[] = ("'attribute' in request data must be a valid json dict string,"
        " and has key 'pts'."
        " pts must be in the form as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."
        " all of x1, x2, y1, y2 can be parsed into int values."
        " And also must have (x2>x1 && y2>y1).");

  if (!document.HasMember("detections")){
      return false;
  }
   // need  ignore in the post process
  if (document.HasMember("detections")){
    const auto &ptses = document["detections"];
    if(ptses.Size()==0)
      return false;
    for(int iter=0;iter<ptses.Size();iter++)
    {
      const auto &pts=ptses[iter]["pts"];
      float t_pts[4][2];
      try{
          bool isArray=pts.IsArray();
          if(!isArray){
          return false;
       }
      const int size=pts.Size();
      if(size!=4){
        return false;
       }
      for(int i=0; i<4; i++){
        for(int j=0; j<2; j++){
          t_pts[i][j] = pts[i][j].GetFloat();
        }
       }
      }
      catch (...){
        return false;
      } 

      if(!check_valid_box_pts(t_pts))
      {
        return false;
      }

      BoxF box;
      box.xmin = t_pts[0][0];
      box.ymin = t_pts[0][1];
      box.xmax = t_pts[2][0];
      box.ymax = t_pts[2][1];
      if(box.xmin>=0&&box.xmin<imsize.width&&box.ymin>=0&&box.ymin<imsize.height&&box.xmax>=0&&box.xmax<imsize.width&&box.ymax>=0&&box.ymax<imsize.height){
	boxes->push_back(box);
      }else{
       LOG(WARNING) << "Exceed img bounder!";
       return false;
      }
    }
     return true;
  }
  return false;
}

/**
 * This function is used to transform detect struct results to json format
 */
 inline std::string get_landmark_json(
     const TronLandmarkOutput &landmark_output) {
  using namespace rapidjson;
  Document document;
  auto &alloc = document.GetAllocator();
  Value j_landmarks(kObjectType), j_ress(kArrayType);
  for (const auto &res : landmark_output.objects) {
    Value j_res(kObjectType);
    Value pts(kArrayType);
    for(int i=0;i<res.landmarks_x_y_num/2;i++){
      Value pt(kArrayType);
      pt.PushBack(Value(res.landmark[i*2]), alloc).PushBack(Value(res.landmark[i*2+1]), alloc);
      pts.PushBack(pt, alloc);
    }
    j_res.AddMember("landmark", pts, alloc);

    Value pos(kArrayType);
    pos.PushBack(Value(res.aspects[0]), alloc).PushBack(Value(res.aspects[1]), alloc).PushBack(Value(res.aspects[2]), alloc);
    j_res.AddMember("pos",pos,alloc);

    j_ress.PushBack(j_res, alloc);
  }
  j_landmarks.AddMember("landmarks", j_ress, alloc);
  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  j_landmarks.Accept(writer);
  return std::string(buffer.GetString());
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
                                tron::MetaNetParam *meta_net_param) {
  inference::CreateParams create_params;
  bool success = read_proto_from_array(params, params_size, &create_params);
  if (success) {
    *custom_params = create_params.custom_params();
    LOG(INFO) << *custom_params;
    if (create_params.model_files_size() > 0) {
      const auto &model_file = create_params.model_files(0);
      if (model_file.has_body()) {
        const auto *net_data = model_file.body().data();
        auto net_size = model_file.body().size();
        LOG(INFO) << "Model file's body size: " << net_size;
        LOG(INFO) << "Model file's body md5: " << MD5(model_file.body());
        return read_proto_from_array(net_data, net_size, meta_net_param);
      } else {
        LOG(WARNING) << "Model file has no body!";
        return false;
      }
    } else {
      LOG(WARNING) << "CreateParams has no model files!";
      return false;
    }
  } else {
    LOG(WARNING) << "Parsing CreateParams Error!";
    return false;
  }
}

}  // namespace Tron

#endif  // TRON_COMMON_HELPER_HPP
