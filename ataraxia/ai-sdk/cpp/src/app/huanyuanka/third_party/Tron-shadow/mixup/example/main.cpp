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
#include "MixupPluginFactory.hpp"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace Shadow;
using namespace std;

int batchSize = 16;

const int gpuID = 0;     // which gpu to use
    
const int modelNum = 3;

const char *deploy_file1 = "models/coarse_deploy.prototxt";
const char *model_file1 = "models/coarse_weight.caffemodel";
const char *bin_file1 = "data/coarse_weight.bin";
const int batchSize1 = 16;
vector<int> inputShape1{3,255,255};
vector<float> preParam1{103.52, 116.28, 123.675, 0.017};
vector<const char *> OUTPUT_BLOB_NAMES1 = {"prob"};
vector<string> OUTPUT_BLOB_NAMES_Str1 = {"prob"};

const char *deploy_file2 = "models/fine_deploy.prototxt";
const char *model_file2 = "models/fine_weight.caffemodel";
const char *bin_file2 = "data/fine_weight.bin";
const int batchSize2 = 16;
vector<int> inputShape2{3,255,255};
vector<float> preParam2{103.52, 116.28, 123.675, 0.017};
vector<const char *> OUTPUT_BLOB_NAMES2 = {"prob"};
vector<string> OUTPUT_BLOB_NAMES_Str2 = {"prob"};

const char *deploy_file3 = "models/det_deploy.prototxt";
const char *model_file3 = "models/det_weight.caffemodel";
const char *bin_file3 = "data/det_all.bin";
const int batchSize3 = 16;
vector<int> inputShape3{3,320,320};
vector<float> preParam3{103.52, 116.28, 123.675, 0.017};
vector<const char *> OUTPUT_BLOB_NAMES3 = {"odm_loc", "conf_data", "prior_data"};
vector<string> OUTPUT_BLOB_NAMES_Str3 = {"detection_out"};


const string imagePath = "";
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
        int len = strlen(buffer);
        if(buffer[len - 1] == '\n');
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
                     vector<const char*> outputBlobName,
                     int flag)
{
    IHostMemory *gieModelStream{nullptr};
    MixupFactory pluginFactory;
    
    cout << "********serialize begin*******" << endl;
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
    for(int i = 0; i < engine->getNbBindings(); i++){
        Dims dim = engine->getBindingDimensions(i);
        printf("index %d: %d %d %d\n",i, dim.d[0],dim.d[1],dim.d[2]);
    }

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
    FILE *file = fopen(enginFilename, "wb+");
    if (!file)
    {
        cerr << "Can't open file to write GIE model" << endl;
        exit(-1);
    }
    fwrite(&flag, sizeof(int), 1, file);
    fwrite(gieModelStream->data(), 1, gieModelStream->size(), file);
    printf("engine size: %lu\n", gieModelStream->size());
    fclose(file);
    gieModelStream->destroy();

    printf("serialize done\n" );
}

void serializeDetectionOutputModel(int batchSize, const char *engineFilename){
    IHostMemory *gieModelStream{nullptr};
    cout << "********serialize detectionout begin*******" << endl;
    Logger gLogger;
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    const char* blobName[] = {"odm_loc", "conf_data", "prior_data"};
    ITensor *input[3];
    // for(int i = 0; i < 3; i++)
    //     input[i] = network->addInput(blobName[i], DataType::kFLOAT, Dims3{outputDim[i].d[0],outputDim[i].d[1],outputDim[i].d[2]});
    input[0] = network->addInput(blobName[0], DataType::kFLOAT, Dims3{27348, 1, 1});
    input[1] = network->addInput(blobName[1], DataType::kFLOAT, Dims3{88881, 1, 1});
    input[2] = network->addInput(blobName[2], DataType::kFLOAT, Dims3{2, 27348, 1});

    IPluginLayer *pluginlayer = network->addPlugin(input, 3, *(IPlugin *)createSSDDetectionOutputPlugin({true, false, 0, 13, 1000, 500, 0.10000000149, 0.300000011921, CodeTypeSSD::CENTER_SIZE, {0, 1, 2}, false, true}));
    pluginlayer->getOutput(0)->setName("detection_out");
    pluginlayer->setName("detection_out");
    network->markOutput(*pluginlayer->getOutput(0));
    builder->setMaxBatchSize(batchSize);
    builder->setMaxWorkspaceSize(256 << 20);

    printf("get engine\n");
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine->getNbBindings()==4);
    network->destroy();
    gieModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    FILE *file = fopen(engineFilename, "ab+");
    if (!file)
    {
        cerr << "can not open bin file" << endl;
        exit(-1);
    }
    int size = gieModelStream->size();
    //fwrite(&(gieModelStream->size()), sizeof(size_t), 1, file);
    fwrite(gieModelStream->data(), 1, size, file);
    printf("engine size: %lu\n", gieModelStream->size());

    fclose(file);
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

void printResults(vector<vector<float>> results, int wrap){
    printf("size: %lu \nresults:%lu\n", results.size(), results[0].size());
    for(int i = 0; i < results.size(); i++){
        vector<float> &result = results[i];
        for(int j = 0; j < result.size(); j++){
            printf("%f,", result[j]);
            if((j + 1) % wrap == 0)
                printf("\n");
        }
        printf("\n\n");
    }
}

void printDetResults(vector<vector<float>> &results){
    vector<float> dets = results[0];
    for(int i = 0; i < dets.size(); i+=3500){
        for(int j = 0; j < 3500; j+=7){
            if(dets[i + j + 1] == -1.0)
                break;
            for(int k = 0; k < 7 ; k++)
                printf("%f ",dets[i+j+k]);
            printf("\n");
        }
        printf("\n\n");
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc == 2 && atoi(argv[1]))
    {   
        if(atoi(argv[1]) == 1)
            caffeToGIEModel(deploy_file1, model_file1, batchSize1, bin_file1, OUTPUT_BLOB_NAMES1, 0);
        else if(atoi(argv[1]) == 2)
            caffeToGIEModel(deploy_file2, model_file2, batchSize2, bin_file2, OUTPUT_BLOB_NAMES2, 0);
        else if(atoi(argv[1]) == 3){
            caffeToGIEModel(deploy_file3, model_file3, batchSize3, bin_file3, OUTPUT_BLOB_NAMES3, 1);
            serializeDetectionOutputModel(batchSize3, bin_file3);
        }
        return 0;
    }

    vector<vector<char>> models;
    models.resize(modelNum);
    vector<int> modelSize;
    modelSize.resize(modelNum);

    getModel(models, modelSize, 0, bin_file1);
    getModel(models, modelSize, 2, bin_file2);
    getModel(models, modelSize, 1, bin_file3);

    // getModel(models, modelSize, 2, bin_file3);
    

    vector<vector<int>> inputShapes;
    inputShapes.push_back(inputShape1);
    inputShapes.push_back(inputShape3);
    inputShapes.push_back(inputShape2);

    vector<vector<float>> preParams;
    preParams.push_back(preParam1);
    preParams.push_back(preParam3);
    preParams.push_back(preParam2);

    vector<vector<float>> results1;
    results1.resize(OUTPUT_BLOB_NAMES_Str1.size());
    vector<vector<float>> results3;
    results3.resize(OUTPUT_BLOB_NAMES_Str3.size());
    vector<vector<float>> results2;
    results2.resize(OUTPUT_BLOB_NAMES_Str2.size());

    Net *mixNet = createNet(inputShapes, preParams, InterMethod::bilinear);
    
    ShadowStatus status;

    status = mixNet->init(gpuID ,models, modelSize);

    if (status != shadow_status_success)
    {
        cerr << "Init mixNet failed"
             << "\t"
             << "exit code: " << status << endl;
        return -1;
    }

    vector<string> imageName;
    if (!getImageName(listFile, imageName))
    {
        cerr << "Can't open image_list.list file" << endl;
    }
    
    Timer timer0,timer1,timer2;
    int rounds = imageName.size() / batchSize;
    double total_time0 = 0,total_time1 = 0,total_time2 = 0;
    for (int r = 0; r <= rounds; r++)
    {
        vector<cv::Mat> imgs;
        vector<string> attributes;
        size_t size = (r < rounds ? batchSize : imageName.size() % batchSize);
        if(size == 0)
            break;

        for (int n = 0; n < size; n++)
        {
            string name = imageName[r * batchSize + n];
            cv::Mat img = cv::imread(imagePath + name);
            if (!img.data)
            {
                cerr << "Read image " << name << " error, No Data!" << endl;
                continue;
            }
            imgs.push_back(img);
        }
        
        printf("round %d\n", r);
        results1[0].resize(imgs.size() * 7);
        results2[0].resize(imgs.size() * 48);
        results3[0].resize(imgs.size() * 3500);
        timer0.start();
        status = mixNet->predict(imgs, OUTPUT_BLOB_NAMES_Str1,  results1, 0);
        total_time0 += timer0.get_millisecond();
        timer1.start();
        status = mixNet->predict(imgs, OUTPUT_BLOB_NAMES_Str2,  results2, 1);
        total_time1 += timer1.get_millisecond();
        timer2.start();
        status = mixNet->predict(imgs, OUTPUT_BLOB_NAMES_Str3,  results3, 2);
        total_time2 += timer2.get_millisecond();
        //printResults(results1, 7);
        //printResults(results2, 49);
        //printDetResults(results3);

        if (status != shadow_status_success)
        {
            cerr << "MixNet predict error"
                 << "\t"
                 << "exit code: " << status << endl;
            return -1;
        }
        // print result
    }
    cout << "per image time: " << total_time0 / imageName.size() << " ms" << endl;
    cout << "per image time: " << total_time1 / imageName.size() << " ms" << endl;
    cout << "per image time: " << total_time2 / imageName.size() << " ms" << endl;
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
