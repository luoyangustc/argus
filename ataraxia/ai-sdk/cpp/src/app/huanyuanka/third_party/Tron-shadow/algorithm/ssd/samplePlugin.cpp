#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <memory>
#include <string.h>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "util.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 300;
static const int INPUT_W = 300;
//static const int OUTPUT_SIZE = 10;
const int CLASS_NUM = 21;


const char* deploy_file = "tensorrt.prototxt";
const char* model_file = "ssd.caffemodel";
const char* image_path = "cat.jpg";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "detection_out";
const int   batchSize = 1;


static Logger gLogger;

static Profiler gProfiler;

void caffeToGIEModel(const std::string& deployFile,                 // name for caffe prototxt
                     const std::string& modelFile,                  // name for model
                     unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactory* pluginFactory, // factory for plugin layers
                     IHostMemory *&gieModelStream)                  // output stream for the GIE model
{
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);

    //bool fp16 = builder->platformHasFastFp16();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deploy_file,
                                                              model_file,
                                                              *network, DataType ::kFLOAT);

    // specify which tensors are outputs
    network->markOutput(*blobNameToTensor->find(OUTPUT_BLOB_NAME));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(256 << 20);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
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

size_t GetDimSize(DimsCHW &dim){
    return dim.c() * dim.h() * dim.w();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
            outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    DimsCHW dimIn= static_cast<DimsCHW&&>(engine.getBindingDimensions(inputIndex));
    DimsCHW dimOut = static_cast<DimsCHW&&>(engine.getBindingDimensions(outputIndex));

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * GetDimSize(dimIn) *  sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * GetDimSize(dimOut) * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));


    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * GetDimSize(dimIn) * sizeof(float), cudaMemcpyHostToDevice, stream));

    context.enqueue(batchSize, buffers, stream, nullptr);
    
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * GetDimSize(dimOut) * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory{
public:
    //继承自nvcaffeparser1::IPluginFactory
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override {
        if(!strcmp(layerName, "conv4_3_norm")){
            conv4_3_norm = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDNormalizePlugin(weights, false, false, 0.0001),nvPluginDeleter);
            return conv4_3_norm.get();
        }
        else if(!strcmp(layerName, "detection_out")){
            detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDDetectionOutputPlugin({true, false, 0, 21, 400, 200, 0.4, 0.45, CodeTypeSSD::CENTER_SIZE, {0,1,2}, false, true}),nvPluginDeleter);
            return detection_out.get();
        }
        else if(priorboxIDs.find(std::string(layerName)) != priorboxIDs.end()){
            const int i = priorboxIDs[layerName];
            switch(i){
                case 0:{
                    float minSize = 30.0, maxSize = 60.0, aspectRatio[] = {1.0, 2.0};
                    priorboxLayers[i] = std::unique_ptr<INvPlugin,void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, &maxSize, aspectRatio, 1, 1, 2, true, false, {0.1,0.1,0.2,0.2}, 0, 0, 8.0, 8.0, 0.5}),nvPluginDeleter);
                    break;
                }
                case 1:{
                    float minSize = 60.0, maxSize = 111.0, aspectRatio[] = {1.0, 2.0, 3.0};
                    priorboxLayers[i] = std::unique_ptr<INvPlugin,void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, &maxSize, aspectRatio, 1, 1, 3, true, false, {0.1,0.1,0.2,0.2}, 0, 0, 16.0, 16.0, 0.5}),nvPluginDeleter);
                    break;
                }
                
                case 2:{
                    float minSize = 111.0, maxSize = 162.0, aspectRatio[] = {1.0, 2.0, 3.0};
                    priorboxLayers[i] = std::unique_ptr<INvPlugin,void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, &maxSize, aspectRatio, 1, 1, 3, true, false, {0.1,0.1,0.2,0.2}, 0, 0, 32.0, 32.0, 0.5}),nvPluginDeleter);
                    break;
                }
                
                case 3:{
                    float minSize = 162.0, maxSize = 213.0, aspectRatio[] = {1.0, 2.0, 3.0};
                    priorboxLayers[i] = std::unique_ptr<INvPlugin,void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, &maxSize, aspectRatio, 1, 1, 3, true, false, {0.1,0.1,0.2,0.2}, 0, 0, 64.0, 64.0, 0.5}),nvPluginDeleter);
                    break;
                }
                
                case 4:{
                    float minSize = 213.0, maxSize = 264.0, aspectRatio[] = {1.0, 2.0};
                    priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, &maxSize, aspectRatio, 1, 1, 2, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 100.0, 100.0, 0.5}), nvPluginDeleter);
                    break;
                }
                
                case 5:{
                    float minSize = 264.0, maxSize = 315.0, aspectRatio[] = {1.0, 2.0};
                    priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, &maxSize, aspectRatio, 1, 1, 2, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 300.0, 300.0, 0.5}), nvPluginDeleter);
                    break;
                }
            }
            return priorboxLayers[i].get();
        }

        else {
            std::cout << layerName << std::endl;
            assert(0);
            return nullptr;
        }
    };
    //继承自nvinfer1::IPluginFactory
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override{
        assert(isPlugin(layerName));
        if(!strcmp(layerName, "conv4_3_norm")){
            conv4_3_norm = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDNormalizePlugin(serialData,serialLength),nvPluginDeleter);
            return conv4_3_norm.get();
        }
        else if(!strcmp(layerName, "detection_out")){
            detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDDetectionOutputPlugin(serialData,serialLength),nvPluginDeleter);
            return detection_out.get();
        }
        else if(priorboxIDs.find(std::string(layerName)) != priorboxIDs.end()){
            const int i = priorboxIDs[layerName];
            priorboxLayers[i] = std::unique_ptr<INvPlugin,void (*)(INvPlugin *)>(createSSDPriorBoxPlugin(serialData,serialLength),nvPluginDeleter);
            return priorboxLayers[i].get();
        }
        else {
            std::cout << layerName << std::endl;
            assert(0);
            return nullptr;
        }
    }

    bool isPlugin(const char* name) override {
        return (!strcmp(name, "conv4_3_norm")
                || !strcmp(name, "detection_out")
                || priorboxIDs.find(std::string(name)) != priorboxIDs.end()
                );
    }

    void destroyPlugin(){
        for (unsigned i = 0; i < priorboxIDs.size(); ++i){
            priorboxLayers[i].reset();
        }
        conv4_3_norm.reset();
        detection_out.reset();
    }

    void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { ptr->destroy(); }};
    //normalize
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> conv4_3_norm{ nullptr, nvPluginDeleter };
    //DetectionOutput
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> detection_out{ nullptr, nvPluginDeleter };
    //PriorBox
    std::unordered_map<std::string, int> priorboxIDs = {
        std::make_pair("conv4_3_norm_mbox_priorbox", 0),
        std::make_pair("fc7_mbox_priorbox", 1),
        std::make_pair("conv6_2_mbox_priorbox", 2),
        std::make_pair("conv7_2_mbox_priorbox", 3),
        std::make_pair("conv8_2_mbox_priorbox", 4),
        std::make_pair("conv9_2_mbox_priorbox", 5),             
    };
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> priorboxLayers[6]{{nullptr, nvPluginDeleter},{nullptr, nvPluginDeleter}, {nullptr, nvPluginDeleter},{nullptr, nvPluginDeleter},{nullptr, nvPluginDeleter},{nullptr, nvPluginDeleter}};
         
    
    
};

int main(int argc, char** argv)
{
    PluginFactory pluginFactory;
    IHostMemory *gieModelStream{ nullptr };
    std::cout<<"********serialize begin*******"<<std::endl;
    caffeToGIEModel(deploy_file, model_file, batchSize, &pluginFactory, gieModelStream);

    pluginFactory.destroyPlugin();


    std::cout<<"********read picture*******"<<std::endl;
    
	//单张图片测试
	cv::Mat img = cv::imread(image_path, -1);
    int height = 300;
    int width = 300;
    cv::resize(img, img, cv::Size(height, width));
	// 将img进行通道分离后保存在host中的数组中，在将数据拷贝到cuda中
	int data_size = 3 * height * width;
	float test_data[data_size] = {0};
	float mean[3] = {104.0, 117.0, 123.0};
	for(int k = 0; k < 3; k++){
		for(int i = 0; i < height; i++){
			for(int j = 0; j < width; j++){
				test_data[k*height*width + i*height + j] = 
					img.at<cv::Vec3b>(i, j)[k] - mean[k];
			}
		}
	}
    
    
    std::cout << " Data Size  " << data_size << std::endl;
    
    std::cout<<"*******deserialize begin****"<<std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
    IExecutionContext *context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);
    // run inference
    DimsCHW dimOut = static_cast<DimsCHW&&>(engine->getBindingDimensions(engine->getBindingIndex(OUTPUT_BLOB_NAME)));
    size_t outputSize = GetDimSize(dimOut);    
    float result[outputSize];
    std::cout<<"******inference begin******"<<std::endl;
    doInference(*context, test_data, result, batchSize);
    gProfiler.printLayerTimes();

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    pluginFactory.destroyPlugin();
    
    std::cout<<"*****print ans****"<<std::endl;
    if(!strcmp(OUTPUT_BLOB_NAME , "detection_out")){
        for(size_t i = 0; i < outputSize; i += 7){
            if(result[i + 1] == -1)
                break;
            for(int j = 0; j < 7; j++)
                std::cout <<result[i + j]<<" ";
            std::cout<<std::endl;
        }
    }
    else{
        for(size_t i = 0; i < outputSize; i+=100){
            std::cout << result[i] << std::endl;
        }
    }
    std::cout << "output size is: "<< dimOut.c() <<" * "<<dimOut.h()<<" * "<<dimOut.w()<<std::endl;
    gieModelStream->destroy();
}
