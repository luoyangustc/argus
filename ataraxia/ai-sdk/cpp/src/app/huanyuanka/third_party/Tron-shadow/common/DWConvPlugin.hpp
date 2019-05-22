#ifndef __DWCONVPLUGIN_HPP___
#define __DWCONVPLUGIN_HPP___

#include "Common.hpp"
#include "Util.hpp"

namespace Shadow
{

struct KernelInfo
{
  int kernelSize[2];
  int pad[2];
  int stride[2];
};

void DWConvLayer(int, const float *, float *, const float *, KernelInfo, DimsCHW, DimsCHW, bool, const float *);

// Depthwise-Convolution layer Plugin
class DWConvPlugin : public IPlugin
{
public:
  // 序列化阶段的构造函数
  DWConvPlugin(const Weights *weights, int nbWeights, int nbOutputChannels, KernelInfo kernelInfo, bool bias_term);
  // 反序列化阶段的构造函数
  DWConvPlugin(const void *data, size_t size);
  ~DWConvPlugin();

  // *********** 序列化阶段接口 *****************
  // 返回输出blob的个数
  inline int getNbOutputs() const override { return 1; };
  // 返回输出blob的维度
  Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;
  // 设置工作空间大小（目前所看到的都是返回0）
  inline size_t getWorkspaceSize(int) const override { return 0; };
  // 用于序列化阶段的配置
  void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int maxBatchSize) override{};
  // 执行序列化操作，将类中的成员变量与网络权重保存至buffer中
  void serialize(void *buffer) override;
  // 序列化阶段得到序列化buffer的长度
  size_t getSerializationSize() override;

  // ************* 两个阶段都会执行的接口 *******************
  // 为执行插件层运算做准备
  int initialize() override;
  // 释放initialize创建的变量与内存
  inline void terminate() override;

  // ************* 反序列化阶段接口 *******************
  // 执行插件层运算
  int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

  // 卷积核膨胀
  void dalation(int);

private:
  int mNbOutputChannels;
  KernelInfo mKernelInfo;
  bool mBiasTerm;
  Weights mKernelWeights, mBiasWeights;
  DimsCHW mInputShape, mOutputShape;

  void *mDeviceKernel{nullptr};
  void *mDeviceBias{nullptr};

  DataType mDataType{DataType::kFLOAT};
};

} // namespace Shadow
#endif
