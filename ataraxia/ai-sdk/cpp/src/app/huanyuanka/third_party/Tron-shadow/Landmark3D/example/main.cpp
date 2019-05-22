//
//  main.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/3.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "Net.hpp"
#include "MyUtils.hpp"
#include "Util.hpp"
#include "NvUtils.h"
#include "NvInfer.h"
#include "NvUffParser.h"
#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cudnn.h>
#include <vector>
#include <dirent.h>
#include <unistd.h>
#include <dirent.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;
using namespace Shadow;
using namespace cv;
using namespace rapidjson;

const string Image_path = "image";
const string plot_path = "plot_kpt";
const char *result_path = "landmark.json";
const char *INPUT_BLOB_NAME = "Placeholder";
const char *OUTPUT_BLOB_NAME = "resfcn256/Conv2d_transpose_16/Sigmoid";
const char *UFF_MODEL_PATH = "face.pb.uff";

float pre_param[4] = {103.52, 116.28, 123.675, 0.017};
const int gpuID = 0;
const int batch_size = 1; //batchsize必须为1
const int iteration = 1;
const int input_shape[3] = {3, 256, 256};
const string suffix = ".*.jpg";
char engin_file_name[] = "landmark_engin.bin";

vector<string> face_detection_string =
    {
        R"({"detections": [{"index": 1, "score": 0.9999949932098389, "pts": [[125, 26], [188, 26], [188, 109], [125, 109]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9999871253967285, "pts": [[146, 50], [244, 50], [244, 175], [146, 175]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9996997117996216, "pts": [[58, 61], [145, 61], [145, 185], [58, 185]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9998713731765747, "pts": [[96, 125], [237, 125], [237, 324], [96, 324]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9999957084655762, "pts": [[116, 115], [305, 115], [305, 355], [116, 355]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9999991655349731, "pts": [[44, 44], [180, 44], [180, 232], [44, 232]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9999915361404419, "pts": [[126, 109], [342, 109], [342, 355], [126, 355]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9999749660491943, "pts": [[174, 59], [261, 59], [261, 163], [174, 163]], "class": "face"}]})",
        R"({"detections": [{"index": 1, "score": 0.9999929666519165, "pts": [[106, 28], [190, 28], [190, 142], [106, 142]], "class": "face"}]})"};

void tfToTRTModel(const char *INPUT_BLOB_NAME, const char *OUTPUT_BLOB_NAME, const char *UFF_MODEL_PATH, IHostMemory *&trt_model_stream, unsigned int max_batch_size)
{
    Logger g_logger;
    IUffParser *parser = createUffParser();
    try
    {
        parser->registerInput(INPUT_BLOB_NAME, Dims3(input_shape[0], input_shape[1], input_shape[2]), UffInputOrder::kNCHW);
        parser->registerOutput(OUTPUT_BLOB_NAME);
    }
    catch (...)
    {
        cerr << "Parse uff file failed..." << endl;
    }
    IBuilder *builder = createInferBuilder(g_logger);
    INetworkDefinition *network = builder->createNetwork();
    if (!parser->parse(UFF_MODEL_PATH, *network, nvinfer1::DataType::kFLOAT))
    {
        cerr << "fail to parse uff file" << endl;
        exit(1);
    }
    builder->setMaxBatchSize(max_batch_size);
    builder->setMaxWorkspaceSize(256 << 20);
    ICudaEngine *engine;
    try
    {
        engine = builder->buildCudaEngine(*network);
    }
    catch (...)
    {
        cerr << "Build cuda engine failed, please check blob name setted" << endl;
    }
    assert(engine);
    network->destroy();
    parser->destroy();
    trt_model_stream = engine->serialize();
    engine->destroy();
    builder->destroy();
}

int main(int argc, const char *argv[])
{
    if (batch_size != 1)
    {
        cerr << "Batchsize should be 1" << endl;
        exit(-1);
    }

    int engin_size = 0;

    if (argc == 2 && atoi(argv[1]))
    {
        IHostMemory *trt_model_stream{nullptr};
        cout << "********serialize begin*******" << endl;
        tfToTRTModel(INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, UFF_MODEL_PATH, trt_model_stream, batch_size);
        FILE *file = fopen(engin_file_name, "wb+");
        if (!file)
        {
            cerr << "Can't open file to write TRT model" << endl;
            exit(-1);
        }
        engin_size = trt_model_stream->size();
        fwrite(trt_model_stream->data(), 1, engin_size, file);
        fclose(file);
        trt_model_stream->destroy();
    }

    Net *resfcn = createNet(batch_size, input_shape, pre_param);
    auto *model_fp = fopen(engin_file_name, "rb");
    if (!model_fp)
    {
        cerr << "Can not open engine file" << endl;
        exit(-1);
    }
    fseek(model_fp, 0, SEEK_END);
    engin_size = ftell(model_fp);
    rewind(model_fp);
    cout << engin_size << endl;
    void *data = malloc(engin_size);
    if (!data)
    {
        cerr << "Alloc engine model memory error" << endl;
        exit(-1);
    }
    fread(data, 1, engin_size, model_fp);
    fclose(model_fp);

    ShadowStatus status;
    vector<string> files;
    vector<string> split_result;
    string tmp_name;

    status = getAllFiles(Image_path, suffix, files);

    if (status != shadow_status_success)
    {
        cerr << "Get data failed"
             << "\t"
             << "exit code: " << status << endl;
        return -1;
    }
    sort(files.begin(), files.end());

    if (access(result_path, 6) == -1)
    {
        assert(remove(result_path));
    }

    status = resfcn->init(gpuID, data, engin_size);

    if (status != shadow_status_success)
    {
        cerr << "Init Resfcn failed"
             << "\t"
             << "exit code: " << status << endl;
        return -1;
    }
    if (files.size() == 0)
    {
        cerr << "Invalid data" << endl;
        return -1;
    }

    free(data);
    int rounds = files.size() / batch_size;

    ofstream outfile(result_path, ios::app);
    if (!outfile)
    {
        cerr << "Invalid result path" << endl;
        return -1;
    }

    for (int i = 0; i < rounds; i++)
    {
        vector<Mat> imgs;
        vector<string> img_name;
        vector<string> attributes;
        vector<string> results;

        for (int j = 0; j < batch_size; j++)
        {
            Mat img = cv::imread(files[i * batch_size + j]);
            split_result = mySplit(files[i * batch_size + j], "/");
            tmp_name = split_result[split_result.size() - 1];
            img_name.push_back(tmp_name);

            attributes.push_back(face_detection_string[i]);
            if (!img.data)
            {
                cerr << "Read image " << files[i * batch_size + j] << " error, No Data!" << endl;
                continue;
            }
            imgs.push_back(img);
        }

        status = resfcn->predict(imgs, attributes, results);
        if (status != shadow_status_success)
        {
            cerr << "Resfcn predict error..."
                 << "\t"
                 << "exit code: " << status << endl;
            return -1;
        }
        //保存landmark结果为json文件
        outfile << results[batch_size - 1];
        outfile << "\n";

        vector<vector<float>> landmark(68, vector<float>(3, 0));
        status = parseLandmark(results[0], landmark);
        if (status != shadow_status_success)
        {
            cerr << "Parse landmark predict error..."
                 << "\t"
                 << "exit code: " << status << endl;
            return -1;
        }
        //保存landmark画图结果，可用于验证
        plotLandmark(imgs[0], img_name[0], landmark, plot_path);
        cout << img_name[0] << "  end..." << endl;
    }
    outfile.close();
    status = resfcn->destroy();

    if (status != shadow_status_success)
    {
        cerr << "Resfcn destory error"
             << "\t"
             << "exit code: " << status << endl;
        return -1;
    }
    return 0;
}
