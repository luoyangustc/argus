//
//  ResFcn.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef __RESFCN_HPP__
#define __RESFCN_HPP__
#include "Net.hpp"
#include "NvInfer.h"
#include "NvUffParser.h"
#include <stdio.h>
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;
using namespace cv;
namespace Shadow
{
class resFcn : public Net
{
  public:
    resFcn(int batchSize, const int *inputShape, float *preParam, InterMethod interMethod);
    ShadowStatus init(const int gpuID, void *data, const int batchSize);
    ShadowStatus init(const int gpuID, const std::vector<std::vector<char>> data,const std::vector<int> size){};
    ShadowStatus predict(const vector<Mat> &imgs, const vector<std::string> &attributes, vector<string> &results);
    ShadowStatus destroy();

  private:
    ShadowStatus doInference(float *input_data, float *output_data, int batch_size);
    vector<float> preProcess(const vector<Mat> &imgs, vector<string> attribute, vector<Mat> &affine_matrix);
    vector<Mat> postProcess(vector<Mat> &affine_matrix, vector<Mat> &network_out);
    void dealResult(vector<string> &results, vector<Mat> &position_map);

    ICudaEngine *engine;
    IExecutionContext *context;
    IUffParser *parser;
    IRuntime *runtime;
    //初始化参数列表
    int BATCH_SIZE;
    int INPUT_CHANNELS;
    int INPUT_WIDTH;
    int INPUT_HEIGHT;

    int input_index;
    int output_index;

    void *buffers[2];
    int iteration;
    int run_num;
    int resolution;

    string INPUT_BLOB_NAME;
    string OUTPUT_BLOB_NAME;
};

Net *createNet(int batchSize, const int *inputShape, float *preParam, InterMethod method)
{
    resFcn *net = new resFcn(batchSize, inputShape, preParam, method);
    return net;
}

} // namespace Shadow
#endif /* resFcn_hpp */
