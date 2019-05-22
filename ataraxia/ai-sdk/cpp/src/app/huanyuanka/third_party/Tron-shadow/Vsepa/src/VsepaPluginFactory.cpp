#include "VsepaPluginFactory.hpp"

namespace Shadow
{

// ********** PluginFactory functions
nvinfer1::IPlugin *VsepaPluginFactory::createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "slicer_angle_prob"))
    {
        vector<int> slice_points;
        slice_points.push_back(1);
        slice_points.push_back(2);
        slice = shared_ptr<IPlugin>(pluginFactory.get()->createSlicePlugin(1, slice_points, 3));
        return slice.get();
    }
    else
    {
        std::cout << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

nvinfer1::IPlugin *VsepaPluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "slicer_angle_prob"))
    {
        slice = shared_ptr<IPluginExt>(pluginFactory.get()->createSlicePlugin(serialData, serialLength));
        return slice.get();
    }
    else
    {
        std::cout << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

bool VsepaPluginFactory::isPluginExt(const char *layerName)
{
    return !strcmp(layerName, "slicer_angle_prob");
}

void VsepaPluginFactory::destroyPlugin()
{
    slice.reset();
}

} // namespace Shadow
