#ifndef __REFINEDETPLUGINFACTORY_HPP___
#define __REFINEDETPLUGINFACTORY_HPP___

#include <unordered_map>
#include "PluginFactory.hpp"

namespace Shadow
{

class RefineDetPluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
  public:
    //继承自nvcaffeparser1::IPluginFactory
    virtual nvinfer1::IPlugin *createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) override
    {
        if (normalizeIDs.find(std::string(layerName)) != normalizeIDs.end())
        {
            const int i = normalizeIDs[layerName];
            normalizeLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDNormalizePlugin(weights, false, false, 0.0001), nvPluginDeleter);
            return normalizeLayers[i].get();
        }
        else if (!strcmp(layerName, "detection_out"))
        {
            detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDDetectionOutputPlugin({true, false, 0, 81, 1000, 500, 0.01, 0.45, CodeTypeSSD::CENTER_SIZE, {0, 1, 2}, false, true}), nvPluginDeleter);
            return detection_out.get();
        }
        else if (!strcmp(layerName, "conf_data"))
        {
            armConf = std::shared_ptr<IPlugin>(pluginFactory.get()->createArmConfPlugin(0.01));
            return armConf.get();
        }
        else if (!strcmp(layerName, "prior_data"))
        {
            armLoc = std::shared_ptr<IPlugin>(pluginFactory.get()->createArmLocPlugin());
            return armLoc.get();
        }

        else if (priorboxIDs.find(std::string(layerName)) != priorboxIDs.end())
        {
            const int i = priorboxIDs[layerName];
            switch (i)
            {
            case 0:
            {
                float minSize = 32.0, aspectRatio[] = {2.0};
                priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, nullptr, aspectRatio, 1, 0, 1, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 8.0, 8.0, 0.5}), nvPluginDeleter);
                break;
            }

            case 1:
            {
                float minSize = 64.0, aspectRatio[] = {2.0};
                priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, nullptr, aspectRatio, 1, 0, 1, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 16.0, 16.0, 0.5}), nvPluginDeleter);
                break;
            }

            case 2:
            {
                float minSize = 128.0, aspectRatio[] = {2.0};
                priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, nullptr, aspectRatio, 1, 0, 1, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 32.0, 32.0, 0.5}), nvPluginDeleter);
                break;
            }

            case 3:
            {
                float minSize = 256.0, aspectRatio[] = {2.0};
                priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin({&minSize, nullptr, aspectRatio, 1, 0, 1, true, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 64.0, 64.0, 0.5}), nvPluginDeleter);
                break;
            }
            }
            return priorboxLayers[i].get();
        }

        else
        {
            std::cout << layerName << std::endl;
            assert(0);
            return nullptr;
        }
    };
    //继承自nvinfer1::IPluginFactory
    IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));
        if (normalizeIDs.find(std::string(layerName)) != normalizeIDs.end())
        {
            const int i = normalizeIDs[layerName];
            normalizeLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDNormalizePlugin(serialData, serialLength), nvPluginDeleter);
            return normalizeLayers[i].get();
        }
        else if (!strcmp(layerName, "detection_out"))
        {
            detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
            return detection_out.get();
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
        else if (priorboxIDs.find(std::string(layerName)) != priorboxIDs.end())
        {
            const int i = priorboxIDs[layerName];
            priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
            return priorboxLayers[i].get();
        }
        else
        {
            std::cout << layerName << std::endl;
            assert(0);
            return nullptr;
        }
    }

    bool isPlugin(const char *name) override
    {
        return (normalizeIDs.find(std::string(name)) != normalizeIDs.end() || !strcmp(name, "detection_out") || !strcmp(name, "conf_data") || !strcmp(name, "prior_data") || priorboxIDs.find(std::string(name)) != priorboxIDs.end());
    }

    void destroyPlugin()
    {
        for (unsigned i = 0; i < priorboxIDs.size(); ++i)
        {
            priorboxLayers[i].reset();
        }
        for (unsigned i = 0; i < normalizeIDs.size(); ++i)
        {
            normalizeLayers[i].reset();
        }
        detection_out.reset();
        armConf.reset();
        armLoc.reset();
    }

  private:
    std::shared_ptr<PluginFactory> pluginFactory{new PluginFactory()};

    std::shared_ptr<IPlugin> armConf{nullptr};
    std::shared_ptr<IPlugin> armLoc{nullptr};

    void (*nvPluginDeleter)(INvPlugin *){[](INvPlugin *ptr) { ptr->destroy(); }};

    //DetectionOutput
    std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> detection_out{nullptr, nvPluginDeleter};

    //Normalize
    std::unordered_map<std::string, int> normalizeIDs = {
        std::make_pair("conv5_3_norm", 0),
        std::make_pair("conv4_3_norm", 1)};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> normalizeLayers[2]{{nullptr, nvPluginDeleter}, {nullptr, nvPluginDeleter}};

    //PriorBox
    std::unordered_map<std::string, int> priorboxIDs = {
        std::make_pair("conv4_3_norm_mbox_priorbox", 0),
        std::make_pair("conv5_3_norm_mbox_priorbox", 1),
        std::make_pair("fc7_mbox_priorbox", 2),
        std::make_pair("conv6_2_mbox_priorbox", 3)};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> priorboxLayers[4]{{nullptr, nvPluginDeleter}, {nullptr, nvPluginDeleter}, {nullptr, nvPluginDeleter}, {nullptr, nvPluginDeleter}};
};

} // namespace Shadow
#endif
