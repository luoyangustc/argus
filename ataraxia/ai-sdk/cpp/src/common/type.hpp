#ifndef TRON_COMMON_TYPE_HPP
#define TRON_COMMON_TYPE_HPP

#if defined(_WIN32)
#define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <list>
#include <string>
#include <vector>

namespace tron {

const float EPS = 0.000001f;

class Scalar {
 public:
  Scalar() = default;
  Scalar(int r_t, int g_t, int b_t) {
    r = static_cast<unsigned char>(r_t);
    g = static_cast<unsigned char>(g_t);
    b = static_cast<unsigned char>(b_t);
  }
  unsigned char r = 0, g = 0, b = 0;
};

template <class Dtype>
class Point {
 public:
  Point() = default;
  Point(Dtype x_t, Dtype y_t, float score_t = -1)
      : x(x_t), y(y_t), score(score_t) {}
  Point(const Point<int> &p) : x(p.x), y(p.y), score(p.score) {}
  Point(const Point<float> &p) : x(p.x), y(p.y), score(p.score) {}
  Dtype x = 0, y = 0;
  float score = -1;
};

template <class Dtype>
class Rect {
 public:
  Rect() = default;
  Rect(Dtype x_t, Dtype y_t, Dtype w_t, Dtype h_t)
      : x(x_t), y(y_t), w(w_t), h(h_t) {}
  Rect(const Rect<int> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  Rect(const Rect<float> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  Dtype x = 0, y = 0, w = 0, h = 0;
};

template <class Dtype>
class Size {
 public:
  Size() = default;
  Size(Dtype w_t, Dtype h_t) : w(w_t), h(h_t) {}
  Size(const Size<int> &size) : w(size.w), h(size.h) {}
  Size(const Size<float> &size) : w(size.w), h(size.h) {}
  Dtype w = 0, h = 0;
};

template <class Dtype>
class Box {
 public:
  Box() = default;
  Box(Dtype xmin_t, Dtype ymin_t, Dtype xmax_t, Dtype ymax_t)
      : xmin(xmin_t), ymin(ymin_t), xmax(xmax_t), ymax(ymax_t) {}
  Box(const Box<int> &box) { *this = box; }
  Box(const Box<float> &box) { *this = box; }

  Box &operator=(const Box<int> &box) {
    xmin = static_cast<Dtype>(box.xmin);
    ymin = static_cast<Dtype>(box.ymin);
    xmax = static_cast<Dtype>(box.xmax);
    ymax = static_cast<Dtype>(box.ymax);
    score = box.score;
    label = box.label;
    return *this;
  }
  Box &operator=(const Box<float> &box) {
    xmin = static_cast<Dtype>(box.xmin);
    ymin = static_cast<Dtype>(box.ymin);
    xmax = static_cast<Dtype>(box.xmax);
    ymax = static_cast<Dtype>(box.ymax);
    score = box.score;
    label = box.label;
    return *this;
  }

  Rect<int> RectInt() const {
    return Rect<int>(xmin, ymin, xmax - xmin, ymax - ymin);
  }
  Rect<float> RectFloat() const {
    return Rect<float>(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  Dtype xmin = 0, ymin = 0, xmax = 0, ymax = 0;
  float score = 0;
  int label = -1;
};

using PointI = Point<int>;
using PointF = Point<float>;
using RectI = Rect<int>;
using RectF = Rect<float>;
using SizeI = Size<int>;
using SizeF = Size<float>;
using BoxI = Box<int>;
using BoxF = Box<float>;

using VecPointI = std::vector<PointI>;
using VecPointF = std::vector<PointF>;
using VecRectI = std::vector<RectI>;
using VecRectF = std::vector<RectF>;
using VecSizeI = std::vector<SizeI>;
using VecSizeF = std::vector<SizeF>;
using VecBoxI = std::vector<BoxI>;
using VecBoxF = std::vector<BoxF>;

using VecChar = std::vector<char>;
using VecUChar = std::vector<unsigned char>;
using VecBool = std::vector<bool>;
using VecInt = std::vector<int>;
using VecFloat = std::vector<float>;
using VecDouble = std::vector<double>;
using VecString = std::vector<std::string>;
using ListChar = std::list<char>;
using ListUChar = std::list<unsigned char>;
using ListBool = std::list<bool>;
using ListInt = std::list<int>;
using ListFloat = std::list<float>;
using ListDouble = std::list<double>;
using ListString = std::list<std::string>;

}  // namespace tron

#endif  // TRON_COMMON_TYPE_HPP NOLINT