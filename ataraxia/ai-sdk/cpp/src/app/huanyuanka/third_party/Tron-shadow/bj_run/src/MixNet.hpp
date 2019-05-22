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
#include "MixPluginFactory.hpp"
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
} Model;

typedef struct
{
    int imageId;
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    vector<cv::Point2d> points;
    float conf;
    float *features = NULL;
} FaceBox;

class MixNet : public Net
{
public:
  MixNet(int modelNum);
  ShadowStatus init(const int gpuID, void *data, int dataSize);
  ShadowStatus init(const int gpuID, const std::vector<std::vector<char>> data,const std::vector<int> size);
  ShadowStatus predict(const vector<cv::Mat> &imgs, const vector<string> &attributes, vector<string> &results);
  ShadowStatus predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &outputlayer, std::vector<std::vector<float> > &results, int enginIndex = 0){return shadow_status_not_implemented;};

  ShadowStatus predict(const std::vector<std::vector<float>> &imgs, const std::vector<std::string> &outputlayer, std::vector<std::vector<float> > &results, int enginIndex = 0){return shadow_status_not_implemented;};

  ShadowStatus destroy();

private:
  ShadowStatus allocateMemory();
  void processImageGPU(const vector<cv::Mat> &imgs, int model_id, InterMethod method);
  void** getBuffers(int model_id);

  ShadowStatus predicModel(const vector<cv::Mat> &imgs, int model_id, InterMethod method = bilinear);
  void cropFace(vector<cv::Mat> imgs, vector<FaceBox> &faceBoxes, vector<cv::Mat> &faces);
  void dealMixResult(vector<FaceBox> &boxes, vector<vector<int>> &results, int size);
  void dealDet3Result(vector<FaceBox*> &boxes, int size);
  void saveFeature(vector<FaceBox*> &boxes);
  string getResultJson(vector<int> &result, vector<FaceBox*> &boxes);


  MixPluginFactory pluginFactory;
  IRuntime *infer;
  cudaStream_t stream;

  cv::cuda::GpuMat inputGpuMat;

  vector<Model> models;

  
  string bkLabel[48];
  vector<pair<int, int>> originSize;
  float preParams[3][4] = {{103.52, 116.28, 123.675, 0.017},{127.5, 127.5, 127.5, 0.0078125},{127.5, 127.5, 127.5, 0.0078125}};
  const float threshold[8] = {1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 1.0};

};

Net *createNet(int batchSize, const int *inputShape, float *preParam, InterMethod method)
{
    MixNet *net = nullptr;
    return net;
}

Net *createNet(int modelNum, InterMethod method)
{
    MixNet *net = new MixNet(modelNum);
    return net;
}


} // namespace Shadow
#endif
