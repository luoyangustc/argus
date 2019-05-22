#ifndef __RESIZEBIPLUGIN_H___
#define __RESIZEBIPLUGIN_H___

#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;
using namespace std;

class ResizeBiPlugin: public IPlugin
{
public:
    ResizeBiPlugin(int _height, int _weight);
    ResizeBiPlugin(const void *data, size_t size);
    ~ResizeBiPlugin() {}
    
    inline int getNbOutputs() const override { return nbOutputs; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    inline size_t getWorkspaceSize(int) const override { return 0; };
    
    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;
    int initialize() override { return 0; };
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;
    inline void terminate() override{};
    
    size_t getSerializationSize() override;
    void serialize(void *buffer) override;
    template <typename T> void write(char*& buffer, const T& val);
    template <typename T> void read(const char*& buffer, T& val);

protected:
    int nbOutputs;
    vector<int> DimsInputs;
    vector<int> DimsOutputs;
    int height;
    int weight;
    int channels;
};

#endif
