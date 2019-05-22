#ifndef __MOBILENET_HPP__
#define __MOBILENET_HPP__

#include "Util.hpp"
#include "Net.hpp"
#include "FdPluginFactory.hpp"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

#include "document.h"
#include "writer.h"
#include "stringbuffer.h"

#include <vector>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
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
    void *data_gpu;
    DimsCHW dim;
    int index;
    size_t dataNum;
} Info;

typedef struct
{
    float conf;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int quality_category;
    float quality_cls[5];
    int orient_category;

} FdResult;


class FdNet : public Net
{
  public:
    FdNet(int modelNum_, InterMethod method_);
    ShadowStatus init(const int gpuID, void *data, const int size);
    ShadowStatus init(const int gpuID, const std::vector<std::vector<char>> data,const std::vector<int> size);
    ShadowStatus predict(const vector<cv::Mat> &imgs, const std::vector<std::string> &attributes, std::vector<string> &results);
    ShadowStatus predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &outputlayer, std::vector<std::vector<float> > &results, int enginIndex = 0){return shadow_status_not_implemented;};
    ShadowStatus predict(const std::vector<std::vector<float>> &imgs, const std::vector<std::string> &outputlayer, std::vector<std::vector<float> > &results, int enginIndex = 0){return shadow_status_not_implemented;};


    ShadowStatus destroy();

  private:
    FdPluginFactory pluginFactory;

    IRuntime *infer = nullptr;
    vector<ICudaEngine *> engine;
    vector<IExecutionContext *> context;
    vector<Profiler> gProfiler;
    Logger gLogger;
    cudaStream_t stream;

    vector<vector<Info>> fdInputInfo;
    vector<vector<Info>> fdOutputInfo;
    int modelNum;

    float fdPreParams[4]   = {104,   117,   123,   0.0170000009239};
    float onetPreParams[4] = {127.5, 127.5, 127.5, 0.0078125};
    
    cv::cuda::GpuMat inputGpuMat;
    InterMethod interMethod;
    vector<size_t> maxBatchSizes;
    double pre_time = 0, infer_time = 0, post_time = 0;

    void initEngine(const char *engineFilename);
    ShadowStatus allocateMemory();
    void processImageGPU(const vector<cv::Mat> &imgs, int model_id, float *preParams);
    
    std::vector<std::pair<int, int>> originSize;
    void getFaceBox(int size, vector<vector<FdResult> > &faceBoxes);
    void drawResult(const vector<cv::Mat> &imgs, const vector<string> &attributes, vector< vector<FdResult> >& faces);
    void cropFace(vector<cv::Mat> imgs, vector<vector<FdResult> > &faceBoxes, std::vector<cv::Mat> &faces);
    void dealResultOnet(int size, std::vector<std::vector<FdResult>> &faceBoxes, int &i, int &j);
    void predictOnet(const std::vector<cv::Mat> faces);

    void **getBuffers(int model_id);
    string getDetectJson(bool output_quality,bool output_quality_score, const vector<FdResult> detection_output);


};

/* will be delete*/
Net *createNet(int batchsize, int const *shape, float *preParam, InterMethod method)
{
    FdNet *net = nullptr;
    return net;
}

Net *createNet(int modelNum, InterMethod method)
{
    FdNet *net = new FdNet(modelNum, method);
    return net;
}

} // namespace Shadow
#endif
