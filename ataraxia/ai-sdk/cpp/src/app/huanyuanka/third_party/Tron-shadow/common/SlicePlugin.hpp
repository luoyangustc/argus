#ifndef __SLICEPLUGIN_HPP___
#define __SLICEPLUGIN_HPP___

#include "Common.hpp"
#include "Util.hpp"

namespace Shadow
{

void SliceLayer(int, const float *, float *, int, int, int, int);

// Slice layer Plugin
class SlicePlugin : public IPluginExt
{
public:
  // 序列化阶段的构造函数
  SlicePlugin(int _axis, vector<int> _slice_points, int n_outputs = 1);
  // 反序列化阶段的构造函数
  SlicePlugin(const void *data, size_t size);
  ~SlicePlugin() {}

  // *********** 序列化阶段接口 *****************
  // 返回输出blob的个数
  inline int getNbOutputs() const override { return nbOutputs; };
  // 返回输出blob的维度
  Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;
  // 设置工作空间大小（目前所看到的都是返回0）
  inline size_t getWorkspaceSize(int) const override { return 0; };

  // 支持的数据格式类型全精度/半精度
  bool supportsFormat(DataType type, PluginFormat format) const override { return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }
  // 用于序列化阶段的配置
  void configureWithFormat(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;
  // 执行序列化操作，将类中的成员变量与网络权重保存至buffer中
  void serialize(void *buffer) override;
  // 序列化阶段得到序列化buffer的长度
  size_t getSerializationSize() override;

  // ************* 反序列化阶段接口 *******************
  // 为执行插件层运算做准备
  int initialize() override { return 0; };
  // 执行插件层运算
  int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;
  // 释放initialize创建的变量与内存
  inline void terminate() override{};

protected:
  int DimsInput[3];
  int nbOutputs;
  vector<int> DimsOutputs;
  int axis;
  vector<int> slice_points;
};

} // namespace Shadow
#endif
