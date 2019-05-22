#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "Net.hpp"
#include "Util.hpp"
#include "MixPluginFactory.hpp"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace Shadow;
using namespace std;

const int gpuID = 0;     // which gpu to use

const int modelNum = 3;

const char *deploy_file = "data/final_merged.prototxt";
const char *model_file = "data/Final_merged.caffemodel";
const char *bin_file = "data/mix_engin.bin";
const int batchSize = 16;
vector<const char *> OUTPUT_BLOB_NAMES = {"detection_out", "prob_bk", "prob_pulp"};

const char *deploy_file2 = "data/det3.prototxt";
const char *model_file2 = "data/det3.caffemodel";
const char *bin_file2 = "data/onet.bin";
const int batchSize2 = 16;
vector<const char*> OUTPUT_BLOB_NAMES2 = {"conv6-3", "prob1"};

const char *deploy_file3 = "data/model-r18-slim-merge-bn.prototxt";
const char *model_file3 = "data/model-r18-slim-145-merge-bn.caffemodel";
const char *bin_file3 = "data/face-feature-res18.bin";
const int batchSize3 = 16;
vector<const char*> OUTPUT_BLOB_NAMES3 = {"fc1"};


const std::string imagePath = "";
const char *listFile = "data/filelist.txt";       // list of imagename to test

//获取文件里面所有图片的名字
bool getImageName(const char *fileName, vector<string> &imageName)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
        return false;
    char buffer[300];
    while (fgets(buffer, 300, f))
    {
        //去掉换行符
        buffer[strlen(buffer) - 1] = '\0';
        imageName.push_back(string(buffer));
    }
    fclose(f);
    return true;
}

void caffeToGIEModel(const char *deployFile,                 // name for caffe prototxt
                     const char *modelFile,                  // name for model
                     unsigned int maxBatchSize,
                     const char *enginFilename,
                     vector<const char*> outputBlobName)
{
    IHostMemory *gieModelStream{nullptr};
    MixPluginFactory pluginFactory;
    
    std::cout << "********serialize begin*******" << std::endl;
    Logger gLogger;
    IBuilder *builder = createInferBuilder(gLogger);
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);
    // bool fp16 = builder->platformHasFastFp16();
    
    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile, modelFile, *network, nvinfer1::DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto it : outputBlobName)
        network->markOutput(*blobNameToTensor->find(it));
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
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
    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    //pluginFactory.destroyPlugin();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
    FILE *file = fopen(enginFilename, "wb+");
    if (!file)
    {
        cerr << "Can't open file to write GIE model" << endl;
        exit(-1);
    }
    fwrite(gieModelStream->data(), 1, gieModelStream->size(), file);
    fclose(file);
    gieModelStream->destroy();

    printf("serialize done\n" );
}



void getModel(vector<vector<char>> &models, vector<int> &modelSize, int model_id, const char *engineFilename){
    FILE *file = fopen(engineFilename, "rb");
    if (!file)
    {
        cerr << "can not open engine file" << endl;
        exit(-1);
    }
    fseek(file, 0, SEEK_END);
    int enginSize = ftell(file);
    rewind(file);
    models.at(model_id).resize(enginSize);
    fread((void*)models.at(model_id).data(), 1, enginSize, file);
    modelSize.at(model_id) = enginSize;
    fclose(file);
}

int main(int argc, char **argv)
{
    int enginSize = 0;
    if (argc == 2 && atoi(argv[1]))
    {   
        if(atoi(argv[1]) == 1)
            caffeToGIEModel(deploy_file, model_file, batchSize, bin_file, OUTPUT_BLOB_NAMES);
        else if(atoi(argv[1]) == 2)
            caffeToGIEModel(deploy_file2, model_file2, batchSize2, bin_file2, OUTPUT_BLOB_NAMES2);
        else 
            caffeToGIEModel(deploy_file3, model_file3, batchSize3, bin_file3, OUTPUT_BLOB_NAMES3);
        return 0;
    }

    vector<vector<char>> models;
    models.resize(modelNum);
    vector<int> modelSize;
    modelSize.resize(modelNum);

    getModel(models, modelSize, 0, bin_file);
    getModel(models, modelSize, 1, bin_file2);
    getModel(models, modelSize, 2, bin_file3);

    Net *mixNet = createNet(modelNum);

    ShadowStatus status;
    status = mixNet->init(gpuID ,models, modelSize);
    if (status != shadow_status_success)
    {
        cerr << "Init mixNet failed"
             << "\t"
             << "exit code: " << status << endl;
        return -1;
    }

    std::vector<string> imageName;
    vector<string> results;
    if (!getImageName(listFile, imageName))
    {
        cerr << "Can't open image_list.list file" << endl;
    }

    int rounds = imageName.size() / batchSize;
    double total_time = 0;

    for (int r = 0; r <= rounds; r++)
    {
        std::vector<cv::Mat> imgs;
        std::vector<std::string> attributes;
        size_t size = (r < rounds ? batchSize : imageName.size() % batchSize);
        if(size == 0)
            break;

        for (int n = 0; n < size; n++)
        {
            string name = imageName[r * batchSize + n];
            cv::Mat img = cv::imread(imagePath + name, -1);
            if (!img.data)
            {
                cerr << "Read image " << name << " error, No Data!" << endl;
                continue;
            }
            imgs.push_back(img);
        }
        clock_t start = clock();
        status = mixNet->predict(imgs, attributes, results);
        total_time += double(clock() - start) / CLOCKS_PER_SEC * 1000;

        if (status != shadow_status_success)
        {
            cerr << "MixNet predict error"
                 << "\t"
                 << "exit code: " << status << endl;
            return -1;
        }
        // print result

        for (int i = 0; i < results.size(); i++)
        {
            std::cout << results[i] << std::endl;
        }
    }
    cout << "per image time: " << total_time / imageName.size() << " ms" << endl;
    
    status = mixNet->destroy();
    if (status != shadow_status_success)
    {
        cerr << "MixNet destory error"
             << "\t"
             << "exit code: " << status << endl;
        return -1;
    }
    cout << "\nAll tests done \n";
    return 0;

}
