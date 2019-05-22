#ifndef __REFINEDETNET_HPP__
#define __REFINEDETNET_HPP__

#include "Util.hpp"
#include "Net.hpp"
#include "RefineDetPluginFactory.hpp"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace std;

using namespace std;

namespace Shadow
{

//存储输入输出的信息
//data: 存储数据
//dim: 数据维度
//index: buffers中的索引
typedef struct
{
    float *data;
    DimsCHW dim;
    int index;
} info;

class RefineDetNet : public Net
{
  public:
    RefineDetNet(float *preParam_, InterMethod method_) : interMethod(method_), preParams(preParam_) {}
    ShadowStatus init(const int gpuID, void *data, const int size);
    ShadowStatus init(const int gpuID, const std::vector<std::vector<char>> data,const std::vector<int> size){};
    ShadowStatus predict(const vector<cv::Mat> &imgs, const std::vector<std::string> &attributes, std::vector<std::string> &results);
    ShadowStatus destroy();
    void printTime();

  private:
    RefineDetPluginFactory pluginFactory;
    IRuntime *infer = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    Profiler gProfiler;
    Logger gLogger;
    cudaStream_t stream;

    void **buffers = nullptr;
    info inputInfo;
    vector<info> outputInfo;
    std::vector<std::pair<float, float>> originSize;
    float *preParams;
    cv::cuda::GpuMat inputGpuMat;
    InterMethod interMethod;
    size_t count = 0, batchSize = 0;
    double pre_time = 0, infer_time = 0, post_time = 0;

    void initEngine(const char *engineFilename);
    ShadowStatus allocateMemory();
    void processImageGPU(const vector<cv::Mat> &imgs);
    void dealResult();
};

Net *createNet(int b, const int *s, float *preParam, InterMethod method)
{
    RefineDetNet *net = new RefineDetNet(preParam, method);
    return net;
}

} // namespace Shadow
#endif
