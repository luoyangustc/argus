#include "Util.hpp"
#include "Net.hpp"
#include "RefineDetPluginFactory.hpp"

using namespace Shadow;

const char *deployFile = "data/deploy.prototxt";
const char *modelFile = "data/coco_refinedet_vgg16_320x320_final.caffemodel";
const std::string imagePath = "images/";
float preParams[4] = {104, 117, 123, 1};
const int gpuID = 0;

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

void caffeToGIEModel(const char *deployFile,                        // name for caffe prototxt
                     const char *modelFile,                         // name for model
                     unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactory *pluginFactory, // factory for plugin layers
                     IHostMemory *&gieModelStream)                  // output stream for the GIE model
{
    Logger gLogger;
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);

    //bool fp16 = builder->platformHasFastFp16();
    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile, modelFile,
                                                              *network, nvinfer1::DataType ::kFLOAT);

    // specify which tensors are outputs
    std::vector<const char *> OUTPUT_BLOB_NAMES = {"detection_out"};
    for (auto it : OUTPUT_BLOB_NAMES)
        network->markOutput(*blobNameToTensor->find(it));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(256 << 20);
    builder->setHalf2Mode(false);

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

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

int main(int argc, char **argv)
{
    char engineFilename[] = "data/refinedet_engin.bin";
    // 生成engine
    int batchSize = 2;
    if (argc == 2 && atoi(argv[1])) //argv[1]==1 build new engin
    {
        RefineDetPluginFactory pluginFactory;
        IHostMemory *gieModelStream{nullptr};
        std::cout << "********serialize begin*******" << std::endl;
        caffeToGIEModel(deployFile, modelFile, batchSize, &pluginFactory, gieModelStream);

        FILE *file = fopen(engineFilename, "wb+");
        if (!file)
        {
            cerr << "can not open bin file" << endl;
            exit(-1);
        }
        int size = gieModelStream->size();
        fwrite(&size, sizeof(int), 1, file);
        fwrite(gieModelStream->data(), 1, size, file);
        fclose(file);

        gieModelStream->destroy();
    }

    FILE *file = fopen(engineFilename, "rb");
    if (!file)
    {
        cerr << "can not open engine file" << endl;
        exit(-1);
    }
    int size;
    void *data;
    fread(&size, sizeof(int), 1, file);
    data = malloc(size);
    fread(data, 1, size, file);
    fclose(file);

    Net *refineDetNet = createNet(0, nullptr, preParams);
    refineDetNet->init(gpuID, data, size);
    free(data);

    cout << "********read picture*******" << endl;
    vector<string> imageName;
    vector<std::string> attributes;
    std::vector<string> results;

    GetImageName("data/image_list.list", imageName);
    int rounds = imageName.size() / batchSize;
    for (int r = 0; r <= rounds; r++)
    {
        std::vector<cv::Mat> imgs;
        //最后一轮,处理剩余数量不足batchSize的图片
        size_t size = (r < rounds ? batchSize : imageName.size() % batchSize);

        for (size_t n = 0; n < size; n++)
        {
            string name = imageName[r * batchSize + n];
            cout << name << endl;
            cv::Mat img = cv::imread(imagePath + name, -1);
            imgs.push_back(img);
        }
        if (size > 0)
        {
            cout << "round:" << r << " " << size << endl;
            refineDetNet->predict(imgs, attributes, results);
        }
    }
    refineDetNet->destroy();
    return 0;
}
