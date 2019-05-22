#ifndef __MIXNET_H__
#define __MIXNET_H__

#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <memory>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "MixupPluginFactory.hpp"
#include "document.h"
#include "stringbuffer.h"
#include "writer.h"
#include "Net.hpp"
#include "Util.hpp"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace std;

namespace Shadow
{

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
    ICudaEngine *engine;
    IExecutionContext *context;
    Profiler gProfiler;
    vector<Info> inputInfo;
    vector<Info> outputInfo;
    size_t maxBatchsize;
    void **buffers;
} Model;


class MixupNet : public Net
{
public:
  MixupNet(vector<vector<float>> &preParam, InterMethod method = bilinear);
  ShadowStatus init(const int gpuID, const vector<vector<char>> data,const vector<int> size);
  ShadowStatus predict(const vector<cv::Mat> &imgs, const vector<string> &attributes, vector<string> &results) {return shadow_status_not_implemented;};
  ShadowStatus predict(const vector<cv::Mat> &imgs, const vector<string> &outputlayer, vector<vector<float> > &results, int enginIndex = 0);
  ShadowStatus predict(const vector<vector<float>> &imgs, const vector<string> &outputlayer, vector<vector<float> > &results, int enginIndex = 0) {return shadow_status_not_implemented;};
  ShadowStatus destroy();

private:
  ShadowStatus allocateMemory();
  void processImageGPU(const vector<cv::Mat> &imgs, int model_id);
  void** getBuffers(int model_id);
  void resizeModel(int modelNum_);


  MixupFactory pluginFactory;
  IRuntime *infer;
  
  cudaStream_t stream;
  cv::cuda::GpuMat inputGpuMat;

  vector<Model> models;
  vector<vector<float>> preParams;
  InterMethod method;


  int detBinSize = 2084;//detectionoupt layer engin size
  int detIndex = -1;
  int outputIndex = -1;

};


Net *createNet(int modelNum, InterMethod method)
{
    //MixupNet *net = new MixupNet(modelNum);
    return nullptr;
}

Net *createNet(vector<vector<int>> &inputShape, vector<vector<float>> &preParam, InterMethod method){
    MixupNet *net = new MixupNet(preParam, method);
    return net;
}



} // namespace Shadow
#endif
