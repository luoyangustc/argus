#ifndef __PRELUPLUGIN_HPP___
#define __PRELUPLUGIN_HPP___

#include "Common.hpp"
#include "Util.hpp"

namespace Shadow
{
void PreluLayer(const int count, const int channels, const int dim, const float* bottom_data,
  float* top_data, void* mDeviceKernel, const int div_factor);

/*
	Prelu layer 
	My code doesn't channel_shared_ (only one param), 
	that is a case of Leaky ReLU ( you can implement it by nvinfer1::plugin::createPReLUPlugin)
*/
class PreluPlugin : public IPlugin
{
public:
  PreluPlugin(const Weights *weights, int nbWeights);
  PreluPlugin(const char *weightsFile);

  PreluPlugin(const void* buffer, size_t size);
  ~PreluPlugin(); 
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims);
  int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream);
  int getNbOutputs() const override { return 1;};
  void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;
  void serialize(void* buffer) override;
  size_t getSerializationSize() override;
  inline size_t getWorkspaceSize(int) const override { return 0; }
  int initialize() override;
  void terminate() override;
protected:
  int input_c;
  int input_h;
  int input_w;
  int input_count;
  bool channel_shared_ {false};
  Weights mWeights;
  void* mDeviceKernel{nullptr};

private:
  void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
  {
    deviceWeights = copyToDevice(hostBuffer, size);
    hostBuffer += size;
  }

  void* copyToDevice(const void* data, size_t count)
  {
    void* deviceData;
    cudaMalloc(&deviceData, count);
    cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice);
    return deviceData;
  }

  template<typename T> void read(const char*& buffer, T& val)
  {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }

  template<typename T> void write(char*& buffer, const T& val)
  {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  size_t type2size(DataType type) { return sizeof(float); }

  void convertAndCopyToBuffer(char*& buffer, const Weights& weights)
  {
    memcpy(buffer, weights.values, weights.count * type2size(weights.type));
    buffer += weights.count * type2size(weights.type);
  }
};

} // namespace Shadow
#endif

