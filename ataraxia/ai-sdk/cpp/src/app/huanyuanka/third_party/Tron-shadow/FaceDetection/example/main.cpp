#include "Util.hpp"
#include "Net.hpp"
#include "FdPluginFactory.hpp"

using namespace Shadow;

const char *deployFile = "data/refinedet_v0.0.2.prototxt";
const char *modelFile = "data/refinedet_v0.0.2.caffemodel";

// const char *onetdeployFile = "data/det3.prototxt";
// const char *onetModelFile = "data/det3.caffemodel";
const char *onetdeployFile = "data/quality_v0.0.2.prototxt";
const char *onetModelFile = "data/quality_v0.0.2.caffemodel";

const std::string imagePath = "images/";
const std::string resultPath = "results/";
const int gpuID = 7;
const int fdBatchSize = 8;

const char fdEngineFilename[] = "data/fdnet_engin.bin";

const int dptBatchSize = 8;

const char dptEngineFilename[] = "data/dptnet_engin.bin";

const int onetBatchSize = 5;

const char onetEngineFilename[] = "data/onet_engin.bin";

//获取文件里面所有图片的名字
void GetImageName(const char *fileName, vector<string> &imageName)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
    {
        cerr << "can not open image_list file" << endl;
        exit(-1);
    }
    char buffer[300];
    while (fgets(buffer, 300, f))
    {
        //去掉换行符
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        imageName.push_back(string(buffer));
    }
}

void serializeFdModel(const char *deployFile,                        // name for caffe prototxt
                     const char *modelFile,                         // name for model
                     unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
                     const char* engineFilename,
                     std::vector<const char*> OUTPUT_BLOB_NAMES,
                     std::vector<DimsCHW> &outputDim,
                     bool flag)
{
    FdPluginFactory pluginFactory;
    IHostMemory *gieModelStream{nullptr};

    std::cout << "********serialize fd begin*******" << std::endl;
    Logger gLogger;
    IBuilder *builder = createInferBuilder(gLogger);

    INetworkDefinition *network = builder->createNetwork();
    assert(network);
    ICaffeParser *parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    printf("parse begin %s %s\n", deployFile, modelFile);
    //bool fp16 = builder->platformHasFastFp16();
    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile, modelFile,
                                                              *network, nvinfer1::DataType ::kFLOAT);
    printf("parse done\n");
    for (auto it : OUTPUT_BLOB_NAMES)
        network->markOutput(*blobNameToTensor->find(it));
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(256 << 20);

    printf(" buildCudaEngine\n");
    ICudaEngine *engine;
    try
    {
        engine = builder->buildCudaEngine(*network);
    }
    catch (...)
    {
        cerr << "build cuda engine failed, please check blob name setted" << endl;
    }
    assert(engine);
    printf("destroy\n");
    network->destroy();
    parser->destroy();

    gieModelStream = engine->serialize();

    if(flag){
        for(int i = 1; i <= 3; i++){
            outputDim.push_back(static_cast<DimsCHW &&>(engine->getBindingDimensions(i)));
        }
    }
    // odm_loc_dim = static_cast<DimsCHW &&>(engine->getBindingDimensions(1));
    // conf_data_dim = static_cast<DimsCHW &&>(engine->getBindingDimensions(2));
    // prior_data_dim = static_cast<DimsCHW &&>(engine->getBindingDimensions(3));
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();

    FILE *file = fopen(engineFilename, "wb+");
    if (!file)
    {
        cerr << "can not open bin file" << endl;
        exit(-1);
    }
    int size = gieModelStream->size();
    fwrite(gieModelStream->data(), 1, size, file);
    fclose(file);
}


void serializeDetectionOutputModel(int batchSize, const char *engineFilename,vector<DimsCHW> &outputDim){
    IHostMemory *gieModelStream{nullptr};
    std::cout << "********serialize detectionout begin*******" << std::endl;
    Logger gLogger;
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    const char* blobName[] = {"odm_loc", "conf_data", "prior_data"};
    ITensor *input[3];
    for(int i = 0; i < 3; i++)
        input[i] = network->addInput(blobName[i], DataType::kFLOAT, Dims3{outputDim[i].d[0],outputDim[i].d[1],outputDim[i].d[2]});

    IPluginLayer *pluginlayer = network->addPlugin(input, 3, *(IPlugin *)createSSDDetectionOutputPlugin({true, false, 0, 2, 300, 300, 0.6, 0.3, CodeTypeSSD::CENTER_SIZE, {0, 1, 2}, false, true}));
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
    FILE *file = fopen(engineFilename, "wb+");
    if (!file)
    {
        cerr << "can not open bin file" << endl;
        exit(-1);
    }
    int size = gieModelStream->size();
    fwrite(gieModelStream->data(), 1, size, file);
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

int main(int argc, char **argv)
{
    if (argc == 2 && atoi(argv[1])) //argv[1]==1 build new engin
    {
        vector<DimsCHW> outputDim;
        std::vector<const char*> outputFdBlobNames   = {"odm_loc", "conf_data", "prior_data"};
        std::vector<const char*> outputOnetBlobNames = {"softmax", "pose_softmax"};

        //serializeFdModel(onetdeployFile, onetModelFile, onetBatchSize, onetEngineFilename, outputOnetBlobNames, outputDim, false);
        serializeFdModel(deployFile, modelFile, fdBatchSize, fdEngineFilename, outputFdBlobNames, outputDim, true);
        serializeDetectionOutputModel(dptBatchSize,dptEngineFilename,outputDim);
    }

    int modelNum = 3; 
    vector<vector<char>> models;
    models.resize(modelNum);
    vector<int> modelSize;
    modelSize.resize(modelNum);

    getModel(models, modelSize, 0, fdEngineFilename);
    getModel(models, modelSize, 1, dptEngineFilename);
    getModel(models, modelSize, 2, onetEngineFilename);

    Net *fdNet = createNet(modelNum);//Fd net
    fdNet->init(gpuID, models, modelSize);
    cout << "********read picture*******" << endl;
    vector<string> imageName;

    GetImageName("data/image_list.list", imageName);
    int rounds = imageName.size() / fdBatchSize;
    double total_time = 0;

    for (int r = 0; r <= rounds; r++)
    {
        vector<cv::Mat> imgs;
        vector<string> attributes;
        vector<string> results;
        //最后一轮,处理剩余数量不足batchSize的图片
        size_t size = (r < rounds ? fdBatchSize : imageName.size() % fdBatchSize);
        for (size_t n = 0; n < size; n++)
        {
            string name = imageName[r * fdBatchSize + n];
            cv::Mat img = cv::imread(imagePath + name, -1);
            imgs.push_back(img);
            attributes.push_back(resultPath + name);
        }
        std::cout<<"iter: "<<r<<std::endl;
        clock_t start = clock();
        if (size > 0)
        {
            fdNet->predict(imgs, attributes, results);            
        }
        total_time += double(clock() - start) / CLOCKS_PER_SEC * 1000;
        std::cout<<"result:"<<std::endl;
        for(int i = 0;i<results.size();i++)
        {
          std::cout<<results.at(i)<<std::endl;
        }
    }
    cout << "per image time: " << total_time / imageName.size() << " ms" << endl;
    fdNet->destroy();
    return 0;
}
