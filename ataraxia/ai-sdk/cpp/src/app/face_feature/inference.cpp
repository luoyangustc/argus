#include "inference.hpp"

#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "face.hpp"

namespace tron {
namespace ff {

void Inference::Predict(const std::vector<cv::Mat> &im_mats,
                        const std::vector<BoxF> &boxes,
                        const std::vector<int> &mirror_tricks,
                        std::vector<std::vector<float>> *features) {
  std::vector<cv::Mat> aligned_faces;
  std::vector<BoxF> _boxes;
  for (std::size_t i = 0; i < im_mats.size(); i++) {
    auto im_mat = im_mats[i];
    BoxF box = boxes[i];

    const int image_w = im_mat.cols, image_h = im_mat.rows;
    const int w = box.xmax - box.xmin, h = box.ymax - box.ymin;
    int diff = std::abs(w - h);
    if (w > h) {
      box.ymin -= diff / 2;
      box.ymax += diff / 2;
      if (box.ymin < 0) box.ymin = 0;
      if (box.ymax > image_h - 1) box.ymax = image_h - 1;
    } else {
      box.xmin -= diff / 2;
      box.xmax += diff / 2;
      if (box.xmin < 0) box.xmin = 0;
      if (box.xmax > image_w - 1) box.xmax = image_w - 1;
    }
    LOG(INFO) << i << " "
              << box.xmin << "," << box.ymin << ","
              << box.xmax << "," << box.ymax;
    _boxes.push_back(box);
  }

  std::vector<VecPointF> points;
  mtcnn_->Predict(im_mats, _boxes, &points);

  for (std::size_t i = 0; i < im_mats.size(); i++) {
    // Process face alignment
    std::vector<cv::Point2d> point2ds;
    for (auto cur = points[i].begin(); cur != points[i].end(); cur++) {
      point2ds.emplace_back(cur->x, cur->y);
    }
    cv::Mat aligned_face = cv::Mat();
    faceAlignmet(im_mats[i].clone(), point2ds, &aligned_face, FACE_112_112);
    // cv::imwrite("/src/res/tmp/2-1.jpg",
    //             im_mats[i].clone()(cv::Rect(boxes[i].xmin, boxes[i].ymin,
    //                                         boxes[i].xmax - boxes[i].xmin,
    //                                         boxes[i].ymax - boxes[i].ymin)));
    // cv::imwrite("/src/res/tmp/2-2.jpg", aligned_face);
    aligned_faces.push_back(aligned_face);
  }

  std::vector<FeatureRequest> reqs;
  VecRectF rois;
  for (std::size_t i = 0; i < im_mats.size(); i++) {
    reqs.emplace_back(aligned_faces[i],
                      RectF(0, 0,
                            aligned_faces[i].cols,
                            aligned_faces[i].rows),
                      mirror_tricks[i]);
    // rois.push_back(RectF(boxes[i].xmin, boxes[i].ymin,
    //                      boxes[i].xmax - boxes[i].xmin,
    //                      boxes[i].ymax - boxes[i].ymin));
  }
  feature_->Predict(reqs, features);
  // feature_->Predict(im_mats, rois, mirror_tricks, features);
}

}  // namespace ff
}  // namespace tron
