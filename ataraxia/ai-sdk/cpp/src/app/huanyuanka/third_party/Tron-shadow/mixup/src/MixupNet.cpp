#include "MathFunction.hpp"
#include "MixupNet.hpp"
#include <time.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace rapidjson;

namespace Shadow
{

static Logger gLogger;
static Profiler gProfiler;

inline size_t getDimSize(DimsCHW &dim)
{
    return dim.c() * dim.h() * dim.w();
}


MixupNet::MixupNet(vector<vector<float>> &preParam, InterMethod method)
{
    for(int i = 0;i < preParam.size(); i++)
    {
        this->preParams.push_back(preParam.at(i));
    }
    this->method = method;
}

void MixupNet::resizeModel(int modelNum_)
{
    models.resize(modelNum_);
    for (size_t i = 0; i<modelNum_; i++)
    {
        models[i].engine = nullptr;
        models[i].context = nullptr;
        models[i].maxBatchsize = 0;
        models[i].buffers = nullptr;
    }
}

ShadowStatus MixupNet::init(const int gpuID, const std::vector<std::vector<char>> data,const std::vector<int> size)
{
    if(this->preParams.size() != data.size() || this->preParams.size() != size.size())
    {
        std::cout<<"Please insure <preParams>, <models>, <model size>  which size are equal"<<std::endl;
        return shadow_status_initpara_error;
    }
    int modelNum_ = data.size();
    vector<const char*> data_p;
    vector<int> new_sizes;
    for(size_t i = 0; i < data.size(); i++)
    {
        if(*(int *)data[i].data() == 1)
        {
            detIndex = i;
            outputIndex = data.size();
            modelNum_++;
        }
        data_p.push_back(&data[i][sizeof(int)]);
        new_sizes.push_back(size[i] - sizeof(int));
    }
    if(detIndex != -1)
    {
        new_sizes[detIndex] = new_sizes[detIndex] - detBinSize;
        new_sizes.push_back(detBinSize);
        data_p.push_back(&data[detIndex][new_sizes[detIndex] + sizeof(int)]);
    }

    resizeModel(modelNum_);

    if (data.size() == 0 || size.size() == 0)
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
        for(size_t i = 0; i < models.size(); i++)
        {
            models[i].engine  = infer->deserializeCudaEngine(data_p[i], new_sizes[i], &pluginFactory);
            models[i].context = models[i].engine->createExecutionContext();
            models[i].context->setProfiler(&(models[i].gProfiler));
            models[i].maxBatchsize = models[i].engine->getMaxBatchSize();
        }
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


ShadowStatus MixupNet::allocateMemory()
{   
    for(size_t model_id = 0; model_id < models.size(); model_id++)
    {
        Model &model = models[model_id];
        int bindings = model.engine->getNbBindings();
        for (int i = 0; i < bindings; i++)
        {
            Info info;
            info.index = i;
            info.dim = static_cast<DimsCHW &&>(model.engine->getBindingDimensions(i));
            info.dataNum = getDimSize(info.dim);
            size_t memory = model.maxBatchsize * info.dataNum * sizeof(float);//using fd index 0 ,malloc bathsize memory for detectionputengin input
            try
            {
                info.data = (float *)malloc(memory);
                assert(info.data);
            }
            catch (...)
            {
                return shadow_status_host_malloc_error;
            }
            try
            {
                cudaMalloc(&info.data_gpu, memory);
                assert(info.data_gpu);
            }
            catch (...)
            {
                return shadow_status_cuda_malloc_error;
            }
            
            if (model.engine->bindingIsInput(i))
            {
                //input
                model.inputInfo.push_back(info);
            }
            else
            {
                //output
                model.outputInfo.push_back(info);
            }
        }
        model.buffers = getBuffers(model_id);
    }
    return shadow_status_success;
}

void MixupNet::processImageGPU(const vector<cv::Mat> &imgs, int model_id)
{
    void *data_gpu = models[model_id].inputInfo[0].data_gpu;
    DimsCHW dim = models[model_id].inputInfo[0].dim;
    size_t dataNum = models[model_id].inputInfo[0].dataNum;
    for (size_t b = 0; b < imgs.size(); b++)
    {
        inputGpuMat.upload(imgs[b]);
        process(inputGpuMat, (float *)(data_gpu) + b * dataNum,
                dim.w(), dim.h(), &preParams[model_id][0], method);
    }
}

void **MixupNet::getBuffers(int model_id)
{
    int n = models[model_id].engine->getNbBindings();
    vector<Info> &input = models[model_id].inputInfo;
    vector<Info> &output = models[model_id].outputInfo;
    void **buffers = (void**)malloc(sizeof(void*) * n);
    for(int i = 0; i < input.size(); i++)
    {
        buffers[input[i].index] = input[i].data_gpu;
    }
    for(int i = 0; i < output.size(); i++)
    {
        buffers[output[i].index] = output[i].data_gpu;
    }
    return buffers;
}

ShadowStatus MixupNet::predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &outputlayer, std::vector<std::vector<float> > &results,int enginIndex)
{

    Model &model = models[enginIndex];
    vector<Info> &infos = model.outputInfo;

    if (imgs.size() == 0)
    {
        return shadow_status_batchsize_zero_error;
    }
    if (imgs.size() > model.maxBatchsize)
    {
        cout << "Please input the valid number of images: max = " << model.maxBatchsize << endl;
        return shadow_status_batchsize_exceed_error;
    }

    processImageGPU(imgs, enginIndex);

    void **buffers = models[enginIndex].buffers;
    model.context->enqueue(imgs.size(), buffers, stream, nullptr);


    if(enginIndex == detIndex){
        Model &det = models[outputIndex];
        buffers = models[outputIndex].buffers;
        for(size_t i = 0; i < imgs.size(); i++)
        {   
            buffers[det.inputInfo[0].index]  = (float*)infos[0].data_gpu + i * infos[0].dataNum;
            buffers[det.inputInfo[1].index]  = (float*)infos[1].data_gpu + i * infos[1].dataNum;
            buffers[det.inputInfo[2].index]  = (float*)infos[2].data_gpu + i * infos[2].dataNum;
            buffers[det.outputInfo[0].index] = (float*)det.outputInfo[0].data_gpu + i * det.outputInfo[0].dataNum;
            det.context->enqueue(1, buffers, stream, nullptr);
        }
    }
    const char *detoutputlayer = models[outputIndex].engine->getBindingName(3);
    
    for(size_t i = 0; i < outputlayer.size(); i++)
    {
        Info info;
        if(outputlayer[i] == detoutputlayer)
        {
            info = models[outputIndex].outputInfo[0];
        }
        else
        {
            //根据layer name获取model中的index,即Info中的index字段
            int index = model.engine->getBindingIndex(outputlayer[i].c_str());
            if(index < 0 || model.engine->bindingIsInput(index))
            {
                return shadow_status_layername_error;
            }
            //将该index与model.outputinfo[j].index进行匹配，找到对应的info
            for(int j = 0; j < infos.size(); j++)
            {
                if(infos[j].index == index)
                {
                    index = j;
                    break;
                }
            }
            info = infos[index];
        }
        //printf("output index: %d imgs size : %d  datanum: %d\n", index, imgs.size(), infos[index].dataNum);
        if(results[i].size() < imgs.size() * info.dataNum)
        {
            return shadow_status_results_size_error;
        }
        try
        {
            cudaMemcpyAsync((void*)&results[i][0], info.data_gpu, imgs.size() * info.dataNum * sizeof(float), cudaMemcpyDeviceToHost);
        }
        catch (...)
        {
            return shadow_status_cuda_memcpy_error;
        }

    }

    return shadow_status_success;
}

ShadowStatus MixupNet::destroy()
{
    for(size_t model_id = 0; model_id < models.size(); model_id++)
    {
        Model &model = models[model_id];
        for (size_t i = 0; i < model.inputInfo.size(); i++)
        {
            free(model.inputInfo[i].data);
            cudaFree(model.inputInfo[i].data_gpu);
        }
        for (size_t i = 0; i < model.outputInfo.size(); i++)
        {
            free(model.outputInfo[i].data);
            cudaFree(model.outputInfo[i].data_gpu);
        }
        if (model.context != nullptr)
            model.context->destroy();
        if (model.engine != nullptr)
            model.engine->destroy();
        if (model.buffers != nullptr)
            free(model.buffers);
    }
    if (infer != nullptr)
        infer->destroy();
    cudaStreamDestroy(stream);
    pluginFactory.destroyPlugin();
    delete this;
    return shadow_status_success;
}

} // namespace Shadow
