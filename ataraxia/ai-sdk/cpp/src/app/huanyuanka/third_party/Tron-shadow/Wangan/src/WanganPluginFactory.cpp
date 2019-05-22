#include "WanganPluginFactory.hpp"

namespace Shadow
{
// ********** PluginFactory functions
nvinfer1::IPlugin *WanganFactory::createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights)
{
    //printf("%d\n", nbWeights);
    assert(isPlugin(layerName));
    if (normalizeIDs.find(std::string(layerName)) != normalizeIDs.end())
    {
        const int i = normalizeIDs[layerName];
        normalizeLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDNormalizePlugin(weights, false, false, 0), nvPluginDeleter);
        return normalizeLayers[i].get();
    }

    else if (priorboxIDs.find(std::string(layerName)) != priorboxIDs.end())
    {
        const int i = priorboxIDs[layerName];
        float minSize = minSizes[i];
        float step = 8.0 * pow(2, i);
        float aspectRatio[] = {2.0};
        priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
            createSSDPriorBoxPlugin({&minSize, nullptr, aspectRatio, 1, 0, 1, true, false, {0.10000000149, 0.10000000149, 0.20000000298, 0.20000000298}, 0, 0, step, step, 0.5}), nvPluginDeleter);
        return priorboxLayers[i].get();
    }

    else if (!strcmp(layerName, "conf_data"))
    {
        armConf = std::shared_ptr<IPlugin>(pluginFactory.get()->createArmConfPlugin(0.00999999977648));
        return armConf.get();
    }
    else if (!strcmp(layerName, "prior_data"))
    {
        armLoc = std::shared_ptr<IPlugin>(pluginFactory.get()->createArmLocPlugin());
        return armLoc.get();
    }
    else if (!strcmp(layerName, "detection_out"))
    {
        detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDDetectionOutputPlugin({true, false, 0, 29, 500, 200, 0.20000000298, 0.449999988079, CodeTypeSSD::CENTER_SIZE, {0, 1, 2}, false, true}), nvPluginDeleter);
        return detection_out.get();
    }
    else
    {
        std::cout << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

nvinfer1::IPlugin *WanganFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
    if (normalizeIDs.find(std::string(layerName)) != normalizeIDs.end())
    {
        const int i = normalizeIDs[layerName];
        normalizeLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDNormalizePlugin(serialData, serialLength), nvPluginDeleter);
        return normalizeLayers[i].get();
    }
    else if (priorboxIDs.find(std::string(layerName)) != priorboxIDs.end())
    {
        const int i = priorboxIDs[layerName];
        priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return priorboxLayers[i].get();
    }
    else if (!strcmp(layerName, "conf_data"))
    {
        armConf = std::shared_ptr<IPlugin>(pluginFactory.get()->createArmConfPlugin(serialData, serialLength));
        return armConf.get();
    }
    else if (!strcmp(layerName, "prior_data"))
    {
        armLoc = std::shared_ptr<IPlugin>(pluginFactory.get()->createArmLocPlugin(serialData, serialLength));
        return armLoc.get();
    }
    else if (!strcmp(layerName, "detection_out"))
    {
        detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
        return detection_out.get();
    }
    else
    {
        std::cout << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

// 取消注释可以使用DepthwiseConvolution替代convolution
bool WanganFactory::isPlugin(const char *layerName)
{
    return (normalizeIDs.find(std::string(layerName)) != normalizeIDs.end() ||
            priorboxIDs.find(std::string(layerName)) != priorboxIDs.end() ||
            !strcmp(layerName, "detection_out") ||
            !strcmp(layerName, "conf_data") ||
            !strcmp(layerName, "prior_data"));
}

void WanganFactory::destroyPlugin()
{
    for (unsigned i = 0; i < normalizeIDs.size(); ++i)
    {
        normalizeLayers[i].reset();
    }
    for (unsigned i = 0; i < priorboxIDs.size(); ++i)
    {
        priorboxLayers[i].reset();
    }
    detection_out.reset();
    armConf.reset();
    armLoc.reset();
}

} // namespace Shadow
