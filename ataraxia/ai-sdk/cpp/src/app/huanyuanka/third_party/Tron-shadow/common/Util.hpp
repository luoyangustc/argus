#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <stdio.h>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include "Common.hpp"
#include "MathFunction.hpp"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <chrono>
#include <climits>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <sys/stat.h>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace std;

namespace Shadow
{

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
  public:
    Logger() : Logger(Severity::kWARNING) {}

    Logger(Severity severity) : reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportableSeverity{Severity::kWARNING};
};

class Profiler : public nvinfer1::IProfiler
{
  public:
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;
    virtual void reportLayerTime(const char *layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record &r) { return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime);
    }
};

//time 
inline std::string find_replace(const std::string &str,
                                const std::string &old_str,
                                const std::string &new_str) {
  std::string origin(str);
  size_t pos = 0;
  while ((pos = origin.find(old_str, pos)) != std::string::npos) {
    origin.replace(pos, old_str.length(), new_str);
    pos += new_str.length();
  }
  return origin;
}

class Timer {
 public:
  Timer() {
#if defined(__linux__) || defined(__APPLE__)
    tstart_ = std::chrono::system_clock::now();
#elif defined(_WIN32)
    QueryPerformanceFrequency(&tfrequency_);
    QueryPerformanceCounter(&tstart_);
#endif
  }

  void start() {
#if defined(__linux__) || defined(__APPLE__)
    tstart_ = std::chrono::system_clock::now();
#elif defined(_WIN32)
    QueryPerformanceCounter(&tstart_);
#endif
  }

  double get_microsecond() {
#if defined(__linux__) || defined(__APPLE__)
    tend_ = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(tend_ -
                                                                 tstart_)
        .count();
#elif defined(_WIN32)
    QueryPerformanceCounter(&tend_);
    return 1000000.0 * (tend_.QuadPart - tstart_.QuadPart) /
           tfrequency_.QuadPart;
#endif
  }
  double get_millisecond() { return 0.001 * get_microsecond(); }
  double get_second() { return 0.000001 * get_microsecond(); }

  static tm get_compile_time() {
    int sec, min, hour, day, month, year;
    char s_month[5], month_names[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
    sscanf(__TIME__, "%d:%d:%d", &hour, &min, &sec);
    sscanf(__DATE__, "%s %d %d", s_month, &day, &year);
    month = (strstr(month_names, s_month) - month_names) / 3;
    return tm{sec, min, hour, day, month, year - 1900};
  }

  static std::string get_compile_time_str() {
    tm ltm = get_compile_time();
    return find_replace(std::string(asctime(&ltm)).substr(4), "\n", "");
  }

  static tm get_current_time() {
    time_t now = time(nullptr);
    return *localtime(&now);
  }

  static std::string get_current_time_str() {
    tm ltm = get_current_time();
    return find_replace(std::string(asctime(&ltm)).substr(4), "\n", "");
  }

  static bool is_expired(int year = 0, int mon = 3, int day = 0) {
    tm compile_tm = get_compile_time(), current_tm = get_current_time();
    int year_gap = current_tm.tm_year - compile_tm.tm_year;
    int mon_gap = current_tm.tm_mon - compile_tm.tm_mon;
    int day_gap = current_tm.tm_mday - compile_tm.tm_mday;
    if (year < year_gap) {
      return true;
    } else if (year > year_gap) {
      return false;
    } else {
      if (mon < mon_gap) {
        return true;
      } else if (mon > mon_gap) {
        return false;
      } else {
        return day < day_gap;
      }
    }
  }

 private:
#if defined(__linux__) || defined(__APPLE__)
  std::chrono::time_point<std::chrono::system_clock> tstart_, tend_;
#elif defined(_WIN32)
  LARGE_INTEGER tstart_, tend_, tfrequency_;
#endif
};


template <typename T>
void write(char *&buffer, const T &val)
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char *&buffer, T &val)
{
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
}

template <typename T, typename U>
T bitwise_cast(U u)
{
    return *reinterpret_cast<T *>(&u);
}

float half2float(__half h);
__half float2half(float f);

size_t type2size(DataType type);
void *copyToDevice(const void *data, size_t count);
void convertAndCopyToDevice(void *&deviceWeights, const Weights &weights, DataType mDataType = DataType::kFLOAT);
void convertAndCopyToBuffer(char *&buffer, const Weights &weights, DataType mDataType = DataType::kFLOAT);
void deserializeToDevice(const char *&hostBuffer, void *&deviceWeights, size_t size);


} // namespace Shadow
#endif
