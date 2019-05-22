#include "DWConvPlugin.hpp"

namespace Shadow
{
// ********** SlicePlugin functions

DWConvPlugin::DWConvPlugin(const Weights *weights, int nbWeights, int nbOutputChannels, KernelInfo kernelInfo, bool bias_term)
    : mNbOutputChannels(nbOutputChannels), mKernelInfo(kernelInfo), mBiasTerm(bias_term)
{
    assert(nbWeights <= 2);
    mKernelWeights = weights[0];
    mBiasWeights.values = nullptr;
    mBiasWeights.count = 0;
    mDataType = mKernelWeights.type;

    mKernelWeights.values = malloc(mKernelWeights.count * type2size(mKernelWeights.type));
    memcpy(const_cast<void *>(mKernelWeights.values), weights[0].values, mKernelWeights.count * type2size(mKernelWeights.type));

    if (bias_term)
    {
        mBiasWeights = weights[1];
        assert(mBiasWeights.count == 0 || mBiasWeights.count == mNbOutputChannels);
        assert(mBiasWeights.type == DataType::kFLOAT || mBiasWeights.type == DataType::kHALF);
        mBiasWeights.values = malloc(mBiasWeights.count * type2size(mBiasWeights.type));
        memcpy(const_cast<void *>(mBiasWeights.values), weights[1].values, mBiasWeights.count * type2size(mBiasWeights.type));
    }
}

DWConvPlugin::DWConvPlugin(const void *data, size_t length)
{
    const char *d = static_cast<const char *>(data), *a = d;
    int in_c, in_h, in_w;
    read(d, in_c);
    read(d, in_h);
    read(d, in_w);
    mInputShape = DimsCHW(in_c, in_h, in_w);
    read(d, mNbOutputChannels);
    read(d, mKernelWeights.count);
    read(d, mBiasWeights.count);
    read(d, mKernelInfo.kernelSize[0]);
    read(d, mKernelInfo.kernelSize[1]);
    read(d, mKernelInfo.pad[0]);
    read(d, mKernelInfo.pad[1]);
    read(d, mKernelInfo.stride[0]);
    read(d, mKernelInfo.stride[1]);

    int out_h = (in_h - mKernelInfo.kernelSize[0] + 2 * mKernelInfo.pad[0]) / mKernelInfo.stride[0] + 1;
    int out_w = (in_w - mKernelInfo.kernelSize[1] + 2 * mKernelInfo.pad[1]) / mKernelInfo.stride[1] + 1;
    mOutputShape = DimsCHW(mNbOutputChannels, out_h, out_w);
    mBiasTerm = (mBiasWeights.count != 0);

    mKernelWeights.values = nullptr;
    mBiasWeights.values = nullptr;

    deserializeToDevice(d, mDeviceKernel, mKernelWeights.count * sizeof(float));
    if (mBiasWeights.count)
        deserializeToDevice(d, mDeviceBias, mBiasWeights.count * sizeof(float));
    assert(d == a + length);
}

DWConvPlugin::~DWConvPlugin()
{
    if (mKernelWeights.values)
    {
        free(const_cast<void *>(mKernelWeights.values));
        mKernelWeights.values = nullptr;
    }
    if (mBiasWeights.values)
    {
        free(const_cast<void *>(mBiasWeights.values));
        mBiasWeights.values = nullptr;
    }
}

nvinfer1::Dims DWConvPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    int out_h = (inputs[0].d[1] - mKernelInfo.kernelSize[0] + 2 * mKernelInfo.pad[0]) / mKernelInfo.stride[0] + 1;
    int out_w = (inputs[0].d[2] - mKernelInfo.kernelSize[1] + 2 * mKernelInfo.pad[1]) / mKernelInfo.stride[1] + 1;
    mInputShape = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    mOutputShape = DimsCHW(mNbOutputChannels, out_h, out_w);
    return mOutputShape;
}

int DWConvPlugin::initialize()
{
    if (mKernelWeights.values)
    {
        const int count = mKernelWeights.count * type2size(mKernelWeights.type);
        cudaMalloc(&mDeviceKernel, count);
        cudaMemcpy(mDeviceKernel, mKernelWeights.values, count, cudaMemcpyHostToDevice);
    }
    if (mBiasWeights.values)
        convertAndCopyToDevice(mDeviceBias, mBiasWeights);
    return 0;
}

void DWConvPlugin::terminate()
{
    if (mDeviceKernel)
    {
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
    if (mDeviceBias)
    {
        cudaFree(mDeviceBias);
        mDeviceBias = nullptr;
    }
}

size_t DWConvPlugin::getSerializationSize()
{
    return sizeof(mInputShape.c()) * 3 + sizeof(mNbOutputChannels) +
           sizeof(mKernelWeights.count) + sizeof(mBiasWeights.count) + 6 * sizeof(int) +
           (mKernelWeights.count + mBiasWeights.count) * sizeof(float);
}

void DWConvPlugin::serialize(void *buffer)
{
    char *d = static_cast<char *>(buffer), *a = d;
    write(d, mInputShape.c());
    write(d, mInputShape.h());
    write(d, mInputShape.w());
    write(d, mNbOutputChannels);
    write(d, mKernelWeights.count);
    write(d, mBiasWeights.count);
    write(d, mKernelInfo.kernelSize[0]);
    write(d, mKernelInfo.kernelSize[1]);
    write(d, mKernelInfo.pad[0]);
    write(d, mKernelInfo.pad[1]);
    write(d, mKernelInfo.stride[0]);
    write(d, mKernelInfo.stride[1]);

    convertAndCopyToBuffer(d, mKernelWeights);
    if (mBiasWeights.count)
        convertAndCopyToBuffer(d, mBiasWeights);
    assert(d == a + getSerializationSize());
}

int DWConvPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream)
{
    const float *input_data = reinterpret_cast<const float *>(inputs[0]);
    float *output_data = reinterpret_cast<float *>(outputs[0]);
    float *kernelData = reinterpret_cast<float *>(mDeviceKernel);
    float *biasData = reinterpret_cast<float *>(mDeviceBias);
    DWConvLayer(batchSize, input_data, output_data, kernelData, mKernelInfo, mInputShape, mOutputShape, mBiasTerm, biasData);
}

void DWConvPlugin::dalation(int dal)
{
    mKernelInfo.kernelSize[0] = dal * (mKernelInfo.kernelSize[0] - 1) + 1;
    mKernelInfo.kernelSize[1] = dal * (mKernelInfo.kernelSize[1] - 1) + 1;
}

} // namespace Shadow
