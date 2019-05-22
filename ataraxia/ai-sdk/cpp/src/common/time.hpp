#ifndef TRON_COMMON_TIME_HPP
#define TRON_COMMON_TIME_HPP

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <chrono>  // NOLINT
#include <climits>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

class Time {
 public:
  Time() { set_current(); }

  void set_current() {
#if defined(__linux__) || defined(__APPLE__)
    tnow_ = std::chrono::system_clock::now();
#elif defined(_WIN32)
    QueryPerformanceFrequency(&tfrequency_);
    QueryPerformanceCounter(&tnow_);
#endif
  }

  double since_microsecond(const Time &tm) {
#if defined(__linux__) || defined(__APPLE__)
    return std::chrono::duration_cast<std::chrono::microseconds>(tnow_ -
                                                                 tm.tnow_)
        .count();
#elif defined(_WIN32)
    QueryPerformanceCounter(&tend_);
    return 1000000.0 * (tnow_.QuadPart - tm.tnow_.QuadPart) /
           tfrequency_.QuadPart;
#endif
  }
  double since_microsecond() {
    Time now;
    return since_microsecond(now);
  }

  double since_millisecond(const Time &tm) {
    return 0.001 * since_microsecond(tm);
  }
  double since_millisecond() { return 0.001 * since_microsecond(); }
  double since_second(const Time &tm) {
    return 0.000001 * since_microsecond(tm);
  }
  double since_second() { return 0.000001 * since_microsecond(); }

 private:
#if defined(__linux__) || defined(__APPLE__)
  std::chrono::time_point<std::chrono::system_clock> tnow_;
#elif defined(_WIN32)
  LARGE_INTEGER tnow_, tfrequency_;
#endif
};

#endif  // TRON_COMMON_TIME_HPP NOLINT
