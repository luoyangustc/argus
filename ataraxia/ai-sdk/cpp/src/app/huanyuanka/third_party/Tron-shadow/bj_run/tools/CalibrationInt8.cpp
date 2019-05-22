#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <chrono>
#include <iterator>
#include "Util.hpp"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "BatchStream.h"
#include <time.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace Shadow;

static Logger gLogger;

// stuff we know about the network and the caffe input/output blobs
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const char* gNetworkName{nullptr};
const char int8EngineFileName[] = "data/imagenet_engin_int8.bin";
const char fp32EngineFileName[] = "data/imagenet_engin_fp32.bin";
// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;
    
    for (auto& dir : directories)
    {
        filepath = dir + filepathSuffix;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found) break;
            
            filepath = "../" + filepath; // Try again in parent dir
        }
        
        if (found)
        {
            break;
        }
        
        filepath.clear();
    }
    
    if (filepath.empty())
    {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
                                    [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        throw std::runtime_error("Could not find " + filepathSuffix + " in data directories:\n\t" + directoryList);
    }
    return filepath;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs;
    dirs.push_back(std::string("batches/") + gNetworkName + std::string("/"));
    dirs.push_back(std::string("models/") + gNetworkName + std::string("/"));
    return locateFile(input, dirs);
}
inline double clockDiff(double c1, double c2)
{

	return (c1 - c2) / CLOCKS_PER_SEC * 1000;
}

bool caffeToTRTModel(const std::string& deployFile,		                // name for caffe prototxt
                     const std::string& modelFile,				// name for model
                     const std::vector<std::string>& outputs,			// network outputs
                     unsigned int maxBatchSize,				        // batch size - NB must be at least as large as the batch we want to run with)
                     DataType dataType,
                     IInt8Calibrator* calibrator,
                     nvinfer1::IHostMemory *&trtModelStream)
{
    std::cout<<"Start convert model!"<<std::endl;
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    
    if((dataType == DataType::kINT8 && !builder->platformHasFastInt8()))
        return false;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
                                                              locateFile(modelFile).c_str(),
                                                              *network,dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);
    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setAverageFindIterations(1);
    builder->setMinFindIterations(1);
    //builder->setDebugSync(true);
    builder->setInt8Mode(dataType == DataType::kINT8);
    builder->setInt8Calibrator(calibrator);
    
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);
    
    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    
    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    std::cout<<"Finish convert model!"<<std::endl;
    return true;
}

float doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    float ms{ 0.0f };
   
    double infer_time_start = clock(); 
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
    outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    
    // create GPU buffers and a stream
    Dims3 inputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(INPUT_BLOB_NAME)));
    Dims3 outputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));
    
    size_t inputSize = batchSize*inputDims.d[0]*inputDims.d[1]*inputDims.d[2] * sizeof(float), outputSize = batchSize *
    outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);
    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);
    
    cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    
    cudaMemcpy(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);
    ms += clockDiff(clock(), infer_time_start);
    
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaStreamDestroy(stream);
    return ms;
}


int calculateScore(float* batchProb, float* labels, int batchSize, int outputSize, int threshold)
{
    int success = 0;
    for (int i = 0; i < batchSize; i++)
    {
        float* prob = batchProb + outputSize*i, correct = prob[(int)labels[i]];
        
        int better = 0;
        for (int j = 0; j < outputSize; j++)
            if (prob[j] >= correct)
                better++;
        if (better <= threshold)
            success++;
    }
    return success;
}



class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true)
    : mStream(stream), mReadCache(readCache)
    {
        DimsNCHW dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();
        cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));
        mStream.reset(firstBatch);
    }
    
    virtual ~Int8EntropyCalibrator()
    {
        cudaFree(mDeviceInput);
    }
    
    int getBatchSize() const override { return mStream.getBatchSize(); }
    
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
            return false;
        
        cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice);
        assert(!strcmp(names[0], INPUT_BLOB_NAME));
        bindings[0] = mDeviceInput;
        return true;
    }
    
    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName(), std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        
        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }
    
    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTableName(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
    
private:
    static std::string calibrationTableName()
    {
        assert(gNetworkName);
        
        return std::string("CalibrationTable") + gNetworkName;
    }
    BatchStream mStream;
    bool mReadCache{ true };
    
    size_t mInputCount;
    void* mDeviceInput{ nullptr };
    std::vector<char> mCalibrationCache;
};

std::pair<float, float> scoreModel(int batchSize, int firstBatch, int nbScoreBatches, DataType datatype, IInt8Calibrator* calibrator, bool build, bool quiet = false)
{
    if(build)
    {
    	std::cout<<"Build Engin"<<std::endl;	
        IHostMemory *trtModelStream{ nullptr };
        bool valid = false;
        valid = caffeToTRTModel(std::string(gNetworkName) + ".prototxt", std::string(gNetworkName) + ".caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, batchSize, datatype, calibrator, trtModelStream);

        if(!valid)
        {
          std::cout << "Engine could not be created at this precision" << std::endl;
          return std::pair<float, float>(0,0);
        }

        assert(trtModelStream != nullptr);
        
	std::string engineName;
        if(datatype == DataType::kINT8)
          engineName = fp32EngineFileName;
        else
          engineName = int8EngineFileName;
        FILE *file = fopen(engineName.c_str(), "wb+");
        if (!file)
        {
	    std::cerr << "can not open bin file" << std::endl;
            exit(-1);
        }
        int size = trtModelStream->size();
        fwrite(&size, sizeof(int), 1, file);
        fwrite(trtModelStream->data(), 1, size, file);
        fclose(file);

        trtModelStream->destroy();
     
    }
    
    std::string engineName;
    if(datatype == DataType::kINT8)
        engineName = fp32EngineFileName;
    else
        engineName = int8EngineFileName;
    FILE *file = fopen(engineName.c_str(), "rb");
    if (!file)
    {
        std::cerr << "can not open engine file" << std::endl;
        exit(-1);
    }
    int size;
    void *data;
    fread(&size, sizeof(int), 1, file);
    data = malloc(size);
    fread(data, 1, size, file);
    fclose(file);

    // Create engine and deserialize model.
    IRuntime* infer = createInferRuntime(gLogger);
    assert(infer != nullptr);
    ICudaEngine* engine = infer->deserializeCudaEngine(data, size, nullptr);//if have pulgin, set in ptr
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    free(data);

    BatchStream stream(batchSize, nbScoreBatches);
    stream.skip(firstBatch);
    
    Dims3 outputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));
    int outputSize = outputDims.d[0]*outputDims.d[1]*outputDims.d[2];
    int top1{ 0 }, top5{ 0 };
    float totalTime{ 0.0f };
    std::vector<float> prob(batchSize * outputSize, 0);
    std::cout<<"Inference start!"<<std::endl;    
    while (stream.next())
    {
        totalTime += doInference(*context, stream.getBatch(), &prob[0], batchSize);
        
        top1 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 1);
        top5 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 5);
        
        std::cout << (!quiet && stream.getBatchesRead() % 10 == 0 ? "." : "") << (!quiet && stream.getBatchesRead() % 800 == 0 ? "\n" : "") << std::flush;
    }
    int imagesRead = stream.getBatchesRead()*batchSize;
    float t1 = float(top1) / float(imagesRead), t5 = float(top5) / float(imagesRead);
    
    if (!quiet)
    {
        std::cout << "\nTop1: " << t1 << ", Top5: " << t5 << std::endl;
        std::cout << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and " << totalTime / stream.getBatchesRead() << " ms/batch." << std::endl;
    }
    
    context->destroy();
    engine->destroy();
    infer->destroy();
    return std::make_pair(t1, t5);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Please provide the network as the first argument and build argument." << std::endl;
        exit(0);
    }
    gNetworkName = argv[1];
    bool buildEngin = atoi(argv[2]);
    
    int batchSize = 1, firstScoreBatch = 0, nbScoreBatches = 50000;//推理的batchSize,跳过firstScoreBatch,共推理nbScoreBatches
    CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION;
    
    for (int i = 3; i < argc; i++)
    {
        if (!strncmp(argv[i], "batch=", 6))
            batchSize = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "start=", 6))
            firstScoreBatch = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "score=", 6))
            nbScoreBatches = atoi(argv[i] + 6);
        else
        {
            std::cout << "Unrecognized argument " << argv[i] << std::endl;
            exit(0);
        }
    }
    
    std::cout.precision(6);

    std::cout << "\nFP32 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
    scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kFLOAT, nullptr, buildEngin);
    
    
    std::cout << "\nINT8 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
    BatchStream calibrationStream(1, 5000);//校验每个batchsize为1，共检测5000个batch
    if (calibrationAlgo == CalibrationAlgoType::kENTROPY_CALIBRATION)
    {
        Int8EntropyCalibrator calibrator(calibrationStream, 0);//表示取第0个batch为fistbatch
        scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kINT8, &calibrator, buildEngin);
    }
    
    shutdownProtobufLibrary();
    return 0;
}
