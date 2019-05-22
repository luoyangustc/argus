#ifndef TRON_TERROR_MIXUP_DEBUG_PRINT_HPP
#define TRON_TERROR_MIXUP_DEBUG_PRINT_HPP

#include "glog/logging.h"
#include "pprint11.h"

#define _DEBUG_PRINT_1(x1) \
  LOG(INFO) << __FUNCTION__ << "[DEBUG] " << #x1 << " = " << x1;
#define _DEBUG_PRINT_2(x1, x2)                                                 \
  LOG(INFO) << __FUNCTION__ << "[DEBUG] " << #x1 << " = " << x1 << ", " << #x2 \
            << " = " << x2;
#define _DEBUG_PRINT_3(x1, x2, x3)                                             \
  LOG(INFO) << __FUNCTION__ << "[DEBUG] " << #x1 << " = " << x1 << ", " << #x2 \
            << " = " << x2 << ", " << #x3 << " = " << x3;

#define _DEBUG_PRINT_4(x1, x2, x3, x4)                                         \
  LOG(INFO) << __FUNCTION__ << "[DEBUG] " << #x1 << " = " << x1 << ", " << #x2 \
            << " = " << x2 << ", " << #x3 << " = " << x3 << ", " << #x4        \
            << " = " << x4;
#define _DEBUG_PRINT_5(x1, x2, x3, x4, x5)                                     \
  LOG(INFO) << __FUNCTION__ << "[DEBUG] " << #x1 << " = " << x1 << ", " << #x2 \
            << " = " << x2 << ", " << #x3 << " = " << x3 << ", " << #x4        \
            << " = " << x4 << ", " << #x5 << " = " << x5;

#define _DEBUG_PRINT_GET(_1, _2, _3, _4, _5, NAME, ...) NAME
#define _DEBUG_PRINT(...)                                               \
  _DEBUG_PRINT_GET(__VA_ARGS__, _DEBUG_PRINT_5, _DEBUG_PRINT_4,         \
                   _DEBUG_PRINT_3, _DEBUG_PRINT_2, _DEBUG_PRINT_1, ...) \
  (__VA_ARGS__)

#endif  // TRON_TERROR_MIXUP_DEBUG_PRINT_HPP
