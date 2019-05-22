#ifndef TRON_COMMON_BOXES_HPP
#define TRON_COMMON_BOXES_HPP

#include "type.hpp"

namespace Tron {

namespace Boxes {

template <typename Dtype>
void Clip(const Box<Dtype> &box, Box<Dtype> *clip_box, Dtype min, Dtype max);

template <typename Dtype>
Dtype Size(const Box<Dtype> &box);

template <typename Dtype>
float Intersection(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
float Union(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
float IoU(const Box<Dtype> &box_a, const Box<Dtype> &box_b);

template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<Box<Dtype>> &boxes,
                            float iou_threshold);
template <typename Dtype>
std::vector<Box<Dtype>> NMS(const std::vector<std::vector<Box<Dtype>>> &Gboxes,
                            float iou_threshold);

template <typename Dtype>
void Smooth(const Box<Dtype> &old_box, Box<Dtype> *new_box, float smooth);
template <typename Dtype>
void Smooth(const std::vector<Box<Dtype>> &old_boxes,
            std::vector<Box<Dtype>> *new_boxes, float smooth);

template <typename Dtype>
void Amend(std::vector<std::vector<Box<Dtype>>> *Gboxes, const VecRectF &crops,
           int height = 1, int width = 1);

}  // namespace Boxes

}  // namespace Tron

#endif  // TRON_COMMON_BOXES_HPP
