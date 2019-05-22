#include "RefineDetNet.hpp"
#include <time.h>

using namespace cv;

namespace Shadow
{

inline size_t getDimSize(DimsCHW &dim)
{
    return dim.c() * dim.h() * dim.w();
}

inline double clockDiff(double c1, double c2)
{
    return (c1 - c2) / CLOCKS_PER_SEC * 1000;
}

//获取输入输出相关信息，并分配显存，
ShadowStatus RefineDetNet::allocateMemory()
{
    int n = engine->getNbBindings();
    try
    {
        buffers = (void **)malloc(sizeof(void *) * n);
    }
    catch (...)
    {
        return shadow_status_host_malloc_error;
    }
    for (int i = 0; i < n; i++)
    {
        DimsCHW dim = static_cast<DimsCHW &&>(engine->getBindingDimensions(i));
        size_t memory = batchSize * getDimSize(dim) * sizeof(float);
        float *data;
        try
        {
            data = (float *)malloc(memory);
        }
        catch (...)
        {
            return shadow_status_host_malloc_error;
        }
        try
        {
            cudaMalloc(&buffers[i], memory);
        }
        catch (...)
        {
            return shadow_status_cuda_malloc_error;
        }
        if (engine->bindingIsInput(i))
        {
            printf("Binding %d (%s): Input.\n", i, engine->getBindingName(i));
            inputInfo.index = i;
            inputInfo.dim = dim;
            inputInfo.data = data;
        }
        else
        {
            info output;
            printf("Binding %d (%s): Output.\n", i, engine->getBindingName(i));
            output.index = i;
            output.dim = dim;
            output.data = data;
            outputInfo.push_back(output);
        }
    }
	return shadow_status_success;
}
//初始化网络，
ShadowStatus RefineDetNet::init(const int gpuID, void *data, int size)
{
    if (data == NULL || size == 0)
        return shadow_status_invalid_gie_file;
    try
    {
        cudaSetDevice(gpuID);
    }
    catch (...)
    {
        return shadow_status_set_gpu_error;
    }

    try
    {
        infer = createInferRuntime(gLogger);
        engine = infer->deserializeCudaEngine(data, size, &pluginFactory);
        context = engine->createExecutionContext();
        context->setProfiler(&gProfiler);
        batchSize = engine->getMaxBatchSize();
        originSize.resize(batchSize);
    }
    catch (...)
    {
        return shadow_status_deserialize_error;
    }
    try
    {
        cudaStreamCreate(&stream);
    }
    catch (...)
    {
        shadow_status_create_stream_error;
    }
    return allocateMemory();
}

void RefineDetNet::processImageGPU(const vector<cv::Mat> &imgs)
{
    DimsCHW dim = inputInfo.dim;

    for (size_t b = 0; b < imgs.size(); b++)
    {
        cv::Mat img = imgs[b];
        originSize[b] = make_pair(img.cols, img.rows);
        inputGpuMat.upload(img);
        process(inputGpuMat, (float *)(buffers[inputInfo.index]) + b * getDimSize(dim),
                dim.w(), dim.h(), preParams, interMethod);
    }
}

ShadowStatus RefineDetNet::predict(const vector<cv::Mat> &imgs, const std::vector<std::string> &attributes, std::vector<std::string> &results)
{
    if (batchSize == 0)
    {
        cout << "Please initial the net first" << endl;
        return shadow_status_batchsize_zero_error;
    }
    if (imgs.size() > batchSize || imgs.size() == 0)
    {
        cout << "Please input the valid number of images" << endl;
        return shadow_status_batchsize_exceed_error;
    }
    count += imgs.size();

    double pre_time_start = clock();
    processImageGPU(imgs);
    pre_time += clockDiff(clock(), pre_time_start);

    //infer processing
    double infer_time_start = clock();
    context->enqueue(imgs.size(), buffers, stream, nullptr);
    for (size_t i = 0; i < outputInfo.size(); i++)
    {
        try
        {
            cudaMemcpyAsync(outputInfo[i].data, buffers[outputInfo[i].index],
                            imgs.size() * getDimSize(outputInfo[i].dim) * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        catch (...)
        {
            return shadow_status_cuda_memcpy_error;
        }
    }
    cudaStreamSynchronize(stream);
    infer_time += clockDiff(clock(), infer_time_start);

    //post processing
    double post_time_start = clock();
    dealResult();
    post_time += clockDiff(clock(), post_time_start);
    return shadow_status_success;
}

void RefineDetNet::dealResult()
{
    size_t outputSize = getDimSize(outputInfo[0].dim);
    float *outputData = outputInfo[0].data;
    for (size_t k = 0; k < batchSize; k++)
    {
        for (size_t i = 0; i < outputSize; i += 7)
        {
            if (outputData[k * outputSize + i + 1] == -1)
                break;
            std::cout << k << ":";
            for (int j = 1; j < 7; j++)
            {
                std::cout << outputData[k * outputSize + i + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    gProfiler.printLayerTimes();
}

void RefineDetNet::printTime()
{
    printf("Number of images: %zu\nPre time per image: %f\nInfer time per image: %f\nPost time per image: %f\n",
           count, pre_time / count, infer_time / count, post_time / count);
}

ShadowStatus RefineDetNet::destroy()
{
    // relese the input and output data
    if (inputInfo.data != nullptr)
        free(inputInfo.data);
    for (size_t i = 0; i < outputInfo.size(); i++)
        free(outputInfo[i].data);
    if (buffers != nullptr)
        for (int i = 0; i < engine->getNbBindings(); i++)
        {
            try
            {
                cudaFree(buffers[i]);
            }
            catch (...)
            {
                return shadow_status_cuda_free_error;
            }
        }
    free(buffers);
    cudaStreamDestroy(stream);
    if (context != nullptr)
        context->destroy();
    if (engine != nullptr)
        engine->destroy();
    if (infer != nullptr)
        infer->destroy();
    pluginFactory.destroyPlugin();
    return shadow_status_success;
}

} // namespace Shadow
