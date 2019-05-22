#include "infer_implement.hpp"
#include <math.h>
namespace Tron {

void MobileNetV2::Setup(const tron::MetaNetParam &meta_net_param,
                        const VecInt &in_shape,const int gpu_id) {
  net_.Setup(gpu_id);

  net_.LoadModel(meta_net_param.network(0));

  auto data_shape = net_.GetBlobByName<float>("data")->shape();
  CHECK_EQ(data_shape.size(),4);
  CHECK_EQ(in_shape.size(),1);
  //if (data_shape[0] != in_shape[0]) {
  {
    data_shape[0] = MAX_BATCH_SIZE;//in_shape[0];
    std::map<std::string, VecInt> shape_map;
    shape_map["data"] = data_shape;
    net_.Reshape(shape_map);
  }

  const auto &out_blob = net_.out_blob();
  CHECK_EQ(out_blob.size(),2);

  str_landmarks=out_blob[0];
  str_aspects=out_blob[1];

  batch_ = MAX_BATCH_SIZE;//data_shape[0];
  in_c_ = data_shape[1];
  in_h_ = data_shape[2];
  in_w_ = data_shape[3];
  in_num_ = in_c_ * in_h_ * in_w_;
  in_data_.resize(batch_ * in_num_);

}

void MobileNetV2::Release() {
  net_.Release();
}

void MobileNetV2::Predict(const std::vector<TronMatRect>& im_mat_rect,const VecRectF &rois,
                          TronLandmarkOutput*  GLandmarkAspects) {
  CHECK_EQ(im_mat_rect.size(),rois.size() );
  in_data_.resize(rois.size() * in_num_);
  for(int b = 0; b < rois.size(); ++b){
      ConvertData(im_mat_rect[b].im_mat, in_data_.data() + b * in_num_, rois[b],
      in_c_, in_h_,in_w_);
  }
  Process(in_data_, GLandmarkAspects,im_mat_rect);
}

void MobileNetV2::Process(const VecFloat &in_data,
                          TronLandmarkOutput* GLandmarkAspects,
                           const std::vector<TronMatRect>& im_mat_rect) {
  const int size=im_mat_rect.size();
  int rounds=ceil(float(size)/MAX_BATCH_SIZE);
  for(int round=0;round<rounds;round++)
  { 
    int curr_batch= MAX_BATCH_SIZE;
    if((round+1)==rounds)
       curr_batch=size%curr_batch;
    std::map<std::string, float *> data_map;
    data_map["data"] = const_cast<float *>(in_data.data())+in_num_*round*MAX_BATCH_SIZE;
    std::map<std::string, std::vector<int>> data_shape;
    data_shape["data"] = {curr_batch, in_c_, in_h_, in_w_};
    net_.Reshape(data_shape);
    net_.Forward(data_map);
    const auto *aspects_data   = net_.GetBlobDataByName<float>(str_aspects);
    const auto *landmarks_data = net_.GetBlobDataByName<float>(str_landmarks);
    for(int iter=0;iter<curr_batch;iter++)
    {
     TronLandmarkAspectsPara single;
     int width=im_mat_rect[iter+round*MAX_BATCH_SIZE].face_rect_output.xmax-im_mat_rect[iter+round*MAX_BATCH_SIZE].face_rect_output.xmin+1;
     int height=im_mat_rect[iter+round*MAX_BATCH_SIZE].face_rect_output.ymax-im_mat_rect[iter+round*MAX_BATCH_SIZE].face_rect_output.ymin+1;
     int topx=im_mat_rect[iter+round*MAX_BATCH_SIZE].face_rect_output.xmin;
     int topy=im_mat_rect[iter+round*MAX_BATCH_SIZE].face_rect_output.ymin;
     float* aptr=single.aspects;
     float* lptr=single.landmark;
     const auto* aspects_data_ptr=aspects_data+iter*TronLandmarkAspectsPara::aspects_num;
     const auto* landmarks_data_ptr=landmarks_data+iter*TronLandmarkAspectsPara::landmarks_x_y_num;
     for(int anum=0;anum<TronLandmarkAspectsPara::aspects_num;anum++)
     {
       *(aptr++)=*(aspects_data_ptr++) * 90;//归一化到[-90,90]
     }
     for(int lnum=0;lnum<TronLandmarkAspectsPara::landmarks_x_y_num/2;lnum++)
     {
       *(lptr++)=(*(landmarks_data_ptr++)+landmark_scale[lnum*2])*width+topx;
       *(lptr++)=(*(landmarks_data_ptr++)+landmark_scale[lnum*2+1])*height+topy;
     }
     GLandmarkAspects->objects.push_back(single);
    }
  }
}

}  // namespace Tron
