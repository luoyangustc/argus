#ifndef TRON_EXPORT_INFER_IMPLEMENT_HPP
#define TRON_EXPORT_INFER_IMPLEMENT_HPP

#include "common/boxes.hpp"
#include "common/log.hpp"
#include "common/type.hpp"
#include "common/util.hpp"

#include "core/network.hpp"
#include "proto/tron.pb.h"

#if defined(USE_OpenCV)
#include <opencv2/opencv.hpp>
#endif

namespace Tron {
    
#define MAX_BATCH_SIZE (16)
    
#if defined(USE_OpenCV)
    static inline void ConvertData(const cv::Mat &im_mat, float *data,
                                   const RectF &roi, int channel, int height,
                                   int width, int flag = 1,
                                   bool transpose = false) {
        CHECK(!im_mat.empty());
        CHECK_NOTNULL(data);
        
        int c_ = im_mat.channels(), h_ = im_mat.rows, w_ = im_mat.cols;
        int dst_spatial_dim = height * width;
        
        if (roi.w <= 1 && roi.h <= 1) {
            if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > 1 || roi.y + roi.h > 1) {
                LOG(FATAL) << "Crop region overflow!";
            }
        } else if (roi.w > 1 && roi.h > 1) {
            if (roi.x < 0 || roi.y < 0 || roi.x + roi.w > w_ || roi.y + roi.h > h_) {
                LOG(FATAL) << "Crop region overflow!";
            }
        } else {
            LOG(FATAL) << "Crop scale must be the same!";
        }
        
        float *data_r = nullptr, *data_g = nullptr, *data_b = nullptr,
        *data_gray = nullptr;
        if (channel == 3 && flag == 0) {
            // Convert to RRRGGGBBB
            data_r = data;
            data_g = data + dst_spatial_dim;
            data_b = data + (dst_spatial_dim << 1);
        } else if (channel == 3 && flag == 1) {
            // Convert to BBBGGGRRR
            data_r = data + (dst_spatial_dim << 1);
            data_g = data + dst_spatial_dim;
            data_b = data;
        } else if (channel == 1) {
            // Convert to Gray
            data_gray = data;
        } else {
            LOG(FATAL) << "Unsupported flag " << flag;
        }
        
        auto roi_x = static_cast<int>(roi.w <= 1 ? roi.x * w_ : roi.x);
        auto roi_y = static_cast<int>(roi.h <= 1 ? roi.y * h_ : roi.y);
        auto roi_w = static_cast<int>(roi.w <= 1 ? roi.w * w_ : roi.w);
        auto roi_h = static_cast<int>(roi.h <= 1 ? roi.h * h_ : roi.h);
        
        cv::Rect cv_roi(roi_x, roi_y, roi_w, roi_h);
        cv::Size cv_size(width, height);
        
        cv::Mat im_resize;
        if (roi_x != 0 || roi_y != 0 || roi_w != w_ || roi_h != h_) {
            cv::resize(im_mat(cv_roi), im_resize, cv_size);
        } else {
            cv::resize(im_mat, im_resize, cv_size);
        }
        
        int dst_h = height, dst_w = width;
        if (transpose) {
            cv::transpose(im_resize, im_resize);
            dst_h = width, dst_w = height;
        }
        
        if (channel == 3) {
            CHECK_EQ(c_, 3);
            for (int h = 0; h < dst_h; ++h) {
                const auto *data_src = im_resize.ptr<uchar>(h);
                for (int w = 0; w < dst_w; ++w) {
                    *data_b++ = static_cast<float>(*data_src++);
                    *data_g++ = static_cast<float>(*data_src++);
                    *data_r++ = static_cast<float>(*data_src++);
                }
            }
        } else if (channel == 1) {
            cv::Mat im_gray;
            cv::cvtColor(im_resize, im_gray, cv::COLOR_BGR2GRAY);
            for (int h = 0; h < dst_h; ++h) {
                const auto *data_src = im_gray.ptr<uchar>(h);
                for (int w = 0; w < dst_w; ++w) {
                    *data_gray++ = static_cast<float>(*data_src++);
                }
            }
        } else {
            LOG(FATAL) << "Unsupported flag " << flag;
        }
    }
#endif
    
    class ShadowDetectionRefineDet {
    public:
        ShadowDetectionRefineDet() = default;
        ~ShadowDetectionRefineDet() { Release(); }
        
        void Setup(const tron::MetaNetParam &meta_net_param, const VecInt &in_shape,
                   int gpu_id);
        
#if defined(USE_OpenCV)
        void Predict(const cv::Mat &im_mat, const VecRectF &rois,
                     std::vector<VecBoxF> *Gboxes,
                     std::vector<std::vector<VecPointF>> *Gpoints=nullptr);
#endif
        
        void GetLabels(VecString *labels);
        
        void Release();
        
    private:
        using LabelBBox = std::map<int, VecBoxF>;
        using VecLabelBBox = std::vector<LabelBBox>;
        
        void Process(const VecFloat &in_data, std::vector<VecBoxF> *Gboxes);
        
        void GetLocPredictions(const float *loc_data, int num,
                               int num_preds_per_class, int num_loc_classes,
                               bool share_location, VecLabelBBox *loc_preds);
        void OSGetConfidenceScores(const float *conf_data, const float *arm_conf_data,
                                   int num, int num_preds_per_class, int num_classes,
                                   std::vector<std::map<int, VecFloat>> *conf_preds,
                                   float objectness_score);
        void GetPriorBBoxes(const float *prior_data, int num_priors,
                            VecBoxF *prior_bboxes,
                            std::vector<VecFloat> *prior_variances);
        
        void CasRegDecodeBBoxesAll(const VecLabelBBox &all_loc_preds,
                                   const VecBoxF &prior_bboxes,
                                   const std::vector<VecFloat> &prior_variances,
                                   int num, bool share_location, int num_loc_classes,
                                   int background_label_id,
                                   VecLabelBBox *all_decode_bboxes,
                                   const VecLabelBBox &all_arm_loc_preds);
        void DecodeBBoxes(const VecBoxF &prior_bboxes,
                          const std::vector<VecFloat> &prior_variances,
                          const VecBoxF &bboxes, VecBoxF *decode_bboxes);
        void DecodeBBox(const BoxF &prior_bbox, const VecFloat &prior_variance,
                        const BoxF &bbox, BoxF *decode_bbox);
        
        Shadow::Network net_;
        VecFloat in_data_;
        VecString labels_;
        std::set<int> selected_labels_;
        std::string odm_loc_str_, odm_conf_flatten_str_, arm_priorbox_str_,
        arm_conf_flatten_str_, arm_loc_str_;
        int batch_, in_num_, in_c_, in_h_, in_w_;
        int num_classes_, num_priors_, num_loc_classes_, background_label_id_, top_k_,
        keep_top_k_;
        float threshold_, nms_threshold_, confidence_threshold_, objectness_score_;
        bool share_location_;
    };
    
    
    struct BoxInfo {
        BoxF box;
        float box_reg[4], landmark[10];
    };
    
    using VecBoxInfo = std::vector<BoxInfo>;
    
    class QualityEvalutation {
    public:
        QualityEvalutation() = default;
        ~QualityEvalutation() { Release(); }
        
        void Setup(const tron::MetaNetParam &meta_net_param,
                   const VecInt &in_shape,int gpu_id,float neg_threshold,float pose_threshold,float cover_threshold,float blur_threshold,float quality_threshold);
        
        void Predict(const cv::Mat &im_mat, const VecBoxF &face_boxes,
                     std::vector<Tron::VecPointF>& Gpoints,
                     std::vector<int>& Qalabels);
        
        void Release();
        
    private:
        void Process(const VecFloat &in_data, const VecBoxF &qa_boxes, std::vector<Tron::VecPointF>& Gpoints,
                     std::vector<int>& Qalabels);
        
        void Argmax(const float* data,const int num,int& idx,float& conf);
        
        void GetPts5(const BoxF box,Tron::VecPointF& points,const float* pts5_ptr);
        
        Shadow::Network net_;
        VecFloat thresholds_;
        VecInt data_shape_;
        VecFloat in_data_;
        int batch_, net_in_num_, net_in_c_, net_in_h_, net_in_w_;
        int cls_dim_;
        int pts5_dim_;
        std::string fc6_pts5_str_, fc6_cls_str_;
        const float pts5_mean[10]={31.8,43.6,72.5,43.6,52.2,62.3,34.7,75.5,71.5,75.5};
    };
    
}  // namespace Tron

#endif  // TRON_EXPORT_INFER_IMPLEMENT_HPP
