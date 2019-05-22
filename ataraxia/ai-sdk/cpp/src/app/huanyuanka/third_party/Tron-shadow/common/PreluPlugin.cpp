#include "PreluPlugin.hpp"

namespace Shadow
{

PreluPlugin::PreluPlugin(const Weights *weights, int nbWeights)
{
    //printf("%d\n", nbWeights);
    assert(nbWeights==1);
    mWeights = weights[0];
    //printf("begin parse %s\n", weightsFile);
    //FILE *f = fopen(weightsFile,"r");
    //printf("file open success\n");
    //mWeights.type = DataType::kFLOAT;
    //fscanf(f, "%ld", &mWeights.count);
    //printf("weights num :%ld\n", mWeights.count);
    assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    mWeights.values = malloc(mWeights.count*type2size(mWeights.type));
    //mWeights.values = malloc(mWeights.count*sizeof(float));
    // printf("malloc success\n");
    // float *p = const_cast<float *>(reinterpret_cast<const float*>(mWeights.values));
    // // printf("%p\n", p);
    // // printf("%p\n", mWeights.values);
    // float test;
    // for(int64_t i = 0; i < mWeights.count; i++){
    //     //printf("%lld ", i);
    //     fscanf(f, "%f", p+i);
    //    //printf("%f ", *(p+i));
    // }
    // printf("\n");
    // fclose(f);
    memcpy(const_cast<void*>(mWeights.values), weights[0].values, mWeights.count*type2size(mWeights.type));
}

PreluPlugin::PreluPlugin(const char *weightsFile)
{
    //assert(nbWeights==1);
    //mWeights = weights[0];
    FILE *f = fopen(weightsFile,"r");
    mWeights.type = DataType::kFLOAT;
    fscanf(f, "%ld", &mWeights.count);
    //assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    //mWeights.values = malloc(mWeights.count*type2size(mWeights.type));
    mWeights.values = malloc(mWeights.count*sizeof(float));
    float *p = const_cast<float *>(reinterpret_cast<const float*>(mWeights.values));
    float test;
    for(int64_t i = 0; i < mWeights.count; i++){
        fscanf(f, "%f", p+i);
    }
    fclose(f);
    //memcpy(const_cast<void*>(mWeights.values), weights[0].values, mWeights.count*type2size(mWeights.type));
}

PreluPlugin::PreluPlugin(const void* buffer, size_t size)
{
    const char* d = reinterpret_cast<const char*>(buffer), *a = d;
    read<int>(d,input_c);
    read<int>(d,input_h);
    read<int>(d,input_w);
    read<int>(d,input_count);
    read<bool>(d,channel_shared_);
    read<int64_t>(d,mWeights.count);
    read<DataType>(d,mWeights.type);
    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));//deserializeToDevice(d,mDeviceKernel,mWeights.count);
    memcpy(const_cast<void*>(mWeights.values), d, mWeights.count * type2size(mWeights.type));
    d += mWeights.count * type2size(mWeights.type);
    assert(d == a + size);
}

PreluPlugin::~PreluPlugin()
{   

    if (mWeights.values){
        free(const_cast<void*>(mWeights.values));
    }
}

Dims PreluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}


void PreluPlugin::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int){
    input_c = inputs[0].d[0]; 
    input_h = inputs[0].d[1];
    input_w = inputs[0].d[2];
    input_count = input_c * input_h * input_w;
}

size_t PreluPlugin::getSerializationSize() {
    return 4*sizeof(int) + sizeof(bool) + sizeof(mWeights.count) 
    + sizeof(mWeights.type) +  mWeights.count * type2size(mWeights.type);
}

void PreluPlugin::serialize(void* buffer) {
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, input_c);
    write(d, input_h);
    write(d, input_w);
    write(d, input_count);
    write(d, channel_shared_);
    write(d, mWeights.count);
    write(d, mWeights.type);
    convertAndCopyToBuffer(d,mWeights);
    assert(d == a + getSerializationSize());
}

int PreluPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    const float *bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float *top_data = reinterpret_cast<float*>(outputs[0]);

    const int count = batchSize * input_count;
    const int dim = input_h*input_w;
    const int channels = input_c;
    const int div_factor = channel_shared_ ? channels : 1; //channel_shared_ default is false

    PreluLayer(count,channels,dim,bottom_data,top_data,mDeviceKernel,div_factor);

    return 0;
}

int PreluPlugin::initialize(){
    cudaMalloc(&mDeviceKernel,mWeights.count*type2size(mWeights.type));
    cudaMemcpy(mDeviceKernel,mWeights.values,mWeights.count*type2size(mWeights.type),cudaMemcpyHostToDevice);
    return 0;
}

void PreluPlugin::terminate(){
    if (mDeviceKernel){
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}
}

