#include "SlicePlugin.hpp"

namespace Shadow
{
// ********** SlicePlugin functions

SlicePlugin::SlicePlugin(int _axis, vector<int> _slice_points, int n_outputs)
{
	assert(_axis >= -3 && _axis != 0 && _axis <= 3);
	axis = _axis < 0 ? 4 + _axis : _axis;
	slice_points = _slice_points;
	assert(n_outputs || slice_points.size());
	if (slice_points.size())
	{
		sort(slice_points.begin(), slice_points.end());
		int prev = 0;
		for (auto &i : slice_points)
		{
			assert(i > prev);
			prev = i;
		}
		nbOutputs = slice_points.size() + 1;
	}
	else
	{
		nbOutputs = n_outputs;
	}
}

SlicePlugin::SlicePlugin(const void *data, size_t length)
{
	const char *d = static_cast<const char *>(data);
	read(d, axis);
	read(d, nbOutputs);
	int r;
	for (int i = 0; i < nbOutputs - 1; i++)
	{
		read(d, r);
		slice_points.push_back(r);
	}
	for (int i = 0; i < 3; i++)
	{
		read(d, DimsInput[i]);
	}
	for (int i = 0; i < nbOutputs; i++)
	{
		read(d, r);
		DimsOutputs.push_back(r);
	}
	assert(length == getSerializationSize());
}

nvinfer1::Dims SlicePlugin::getOutputDimensions(int index, const Dims *input, int nbInputDims)
{
	assert(nbInputDims == 1 && input[0].nbDims == 3 && index < nbOutputs);
	if (nbOutputs == 1)
		return nvinfer1::DimsCHW(input[0].d[0], input[0].d[1], input[0].d[2]);
	if (slice_points.size())
	{
		assert(slice_points[index - 1] < input[0].d[axis - 1]);
		if (axis == 1)
		{
			if (index == 0)
				return nvinfer1::DimsCHW(slice_points[0], input[0].d[1], input[0].d[2]);
			else if (index == slice_points.size())
				return nvinfer1::DimsCHW(input[0].d[0] - slice_points[index - 1], input[0].d[1], input[0].d[2]);
			else
				return nvinfer1::DimsCHW(slice_points[index] - slice_points[index - 1], input[0].d[1], input[0].d[2]);
		}
		if (axis == 2)
		{
			if (index == 0)
				return nvinfer1::DimsCHW(input[0].d[0], slice_points[0], input[0].d[2]);
			else if (index == slice_points.size())
				return nvinfer1::DimsCHW(input[0].d[0], input[0].d[1] - slice_points[index - 1], input[0].d[2]);
			else
				return nvinfer1::DimsCHW(input[0].d[0], slice_points[index] - slice_points[index - 1], input[0].d[2]);
		}
		if (axis == 3)
		{
			if (index == 0)
				return nvinfer1::DimsCHW(input[0].d[0], input[0].d[1], slice_points[0]);
			else if (index == slice_points.size())
				return nvinfer1::DimsCHW(input[0].d[0], input[0].d[1], input[0].d[2] - slice_points[index - 1]);
			else
				return nvinfer1::DimsCHW(input[0].d[0], input[0].d[1], slice_points[index] - slice_points[index - 1]);
		}
	}
	else
	{
		assert(input[0].d[axis - 1] % nbOutputs == 0);
		if (axis == 1)
			return DimsCHW(input[0].d[0] / nbOutputs, input[0].d[1], input[0].d[2]);
		if (axis == 2)
			return DimsCHW(input[0].d[0], input[0].d[1] / nbOutputs, input[0].d[2]);
		if (axis == 3)
			return DimsCHW(input[0].d[0], input[0].d[1], input[0].d[2] / nbOutputs);
	}
}

void SlicePlugin::configureWithFormat(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, DataType type, PluginFormat format, int batchSize)
{
	for (int i = 0; i < 3; i++)
		DimsInput[i] = inputs[0].d[i];
	if (slice_points.size())
	{
		int prev = 0;
		for (int i = 0; i < nbOutputs - 1; i++)
		{
			DimsOutputs.push_back(slice_points[i] - prev);
			prev = slice_points[i];
		}
		DimsOutputs.push_back(inputs[0].d[axis - 1] - prev);
	}
	else
	{
		int prev = 0;
		for (int i = 0; i < nbOutputs - 1; i++)
		{
			DimsOutputs.push_back(inputs[0].d[axis - 1] / nbOutputs);
			slice_points.push_back(inputs[0].d[axis - 1] / nbOutputs + prev);
			prev += inputs[0].d[axis - 1];
		}
		DimsOutputs.push_back(inputs[0].d[axis - 1] / nbOutputs);
	}
}

size_t SlicePlugin::getSerializationSize()
{
	// 5个int分别代表 1个axis，1个nbOutputs，3个DimsInput
	return (5 + slice_points.size() + nbOutputs) * sizeof(int);
}

void SlicePlugin::serialize(void *buffer)
{
	char *d = static_cast<char *>(buffer);
	write(d, axis);
	write(d, nbOutputs);
	for (int i = 0; i < slice_points.size(); i++)
	{
		write(d, slice_points[i]);
	}
	for (int i = 0; i < 3; i++)
	{
		write(d, DimsInput[i]);
	}
	for (int i = 0; i < nbOutputs; i++)
	{
		write(d, DimsOutputs[i]);
	}
}

int SlicePlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream)
{
	int num_slices = batchSize;
	for (int i = 0; i < axis - 1; i++)
		num_slices *= DimsInput[i]; // 并行数

	int slice_size = 1;
	for (int i = axis; i < 3; i++)
		slice_size *= DimsInput[i]; // 单位slice的数据长度

	const int input_slice_axis = DimsInput[axis - 1];
	const float *input_data = reinterpret_cast<const float *>(inputs[0]);
	int offset_slice_axis = 0; // 切分数据偏移点

	for (int i = 0; i < nbOutputs; i++)
	{
		const int output_slice_axis = DimsOutputs[i];
		const int output_slice_size = output_slice_axis * slice_size; // 第i个slice的数据长度
		const int nthreads = output_slice_size * num_slices;		  // 线程总数 = 并行数 * 数据长度
		float *output_data = reinterpret_cast<float *>(outputs[i]);
		SliceLayer(nthreads, input_data, output_data, slice_size, input_slice_axis, output_slice_axis, offset_slice_axis);
		offset_slice_axis += output_slice_axis; // 移动切分起始点
	}
}

} // namespace Shadow
