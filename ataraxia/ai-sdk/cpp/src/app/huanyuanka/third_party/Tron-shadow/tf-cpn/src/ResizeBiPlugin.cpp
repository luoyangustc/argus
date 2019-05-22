#include "ResizeBiPlugin.hpp"
#include "MathFunction.hpp"
using namespace std;
ResizeBiPlugin::ResizeBiPlugin(int _height, int _weight)
{
    
    assert(_height > 0 && _weight > 0);
    height = _height;
    weight = _weight;
    channels = 3;
    nbOutputs = 1;
}

ResizeBiPlugin::ResizeBiPlugin(const void *data, size_t length)
{
    const char *d = static_cast<const char *>(data), *a=d;
    read(d, height);
    read(d, weight);
    int tmp;
    for (int i = 0; i < 3; i++)
    {
        read(d, DimsInputs[i]);
    }
    for (int i = 0; i < nbOutputs; i++)
    {
        read(d, DimsOutputs[i]);
    }

    assert(d == a + length);
}


nvinfer1::Dims ResizeBiPlugin::getOutputDimensions(int index, const Dims* input, int nbInputDims) 
{
    assert(index < nbOutputs && nbInputDims == 1 && input[0].nbDims == 3);

    if(input[0].d[1] == height && input[0].d[2] == weight)
        return nvinfer1::DimsCHW(input[0].d[0], input[0].d[1], input[0].d[2]);
    else
        return nvinfer1::DimsCHW(input[0].d[0], height, weight);
}

void ResizeBiPlugin::configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int)
{
    for (int i = 0; i < 3; i++)
        DimsInputs[i] = inputs[0].d[i];
    for (int i = 0; i < nbOutputs; i++)
    {
        DimsOutputs.push_back(1);
        //DimsOutputs.push(height);
        //DimsOutputs.push(weight);
    }
}

size_t ResizeBiPlugin::getSerializationSize()
{
    // 5个int分别代表 1个height，一个weight, 1个nbOutputs，3个DimsInput
    return (5 + nbOutputs) * sizeof(int);
}

void ResizeBiPlugin::serialize(void *buffer)
{
	cout << "start serialize\n";
    char *d = static_cast<char *>(buffer);
    write(d, height);
    write(d, weight);
    for (int i = 0; i < 3; i++)
    {
        write(d, DimsInputs[i]);
    }
    for (int i = 0; i < nbOutputs; i++)
    {
        write(d, DimsOutputs[i]);
    }
}

int ResizeBiPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream)
{
    const float *input_data2 = reinterpret_cast<const float *>(inputs[0]);
	float *input_data = const_cast<float *>(input_data2);
    float *output_data = reinterpret_cast<float *>(outputs[0]);
    float *tmp_in = NULL;
    float *tmp_out = NULL;
   
    vector<float> tmp_data;
    float value,in_height,in_weight;
    in_height = DimsInputs[1];
    in_weight = DimsInputs[2];  
    for(int b = 0; b < batchSize; b++)
    {
        tmp_data.clear();
        tmp_in = &input_data[0] + b * channels * height * weight * sizeof(float);
        tmp_out = &output_data[0] + b * channels * height * weight * sizeof(float);
        
        process(tmp_in, tmp_out, in_weight, in_height, weight, height);
    }
}

template <typename T>
void ResizeBiPlugin::write(char*& buffer, const T& val)
{

	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}

template <typename T>
void ResizeBiPlugin::read(const char*& buffer, T& val)
{

	val = *reinterpret_cast<const T*>(buffer);
	buffer += sizeof(T);
}



