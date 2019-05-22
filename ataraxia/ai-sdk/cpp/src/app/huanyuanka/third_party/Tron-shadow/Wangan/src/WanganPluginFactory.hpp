#ifndef __MOBILEPLUGINFACTORY_HPP___
#define __MOBILEPLUGINFACTORY_HPP___

#include "PluginFactory.hpp"

#define nullPlugin           \
  {                          \
    nullptr, nvPluginDeleter \
  }

namespace Shadow
{
class WanganFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
  virtual IPlugin *createPlugin(const char *layerName, const nvinfer1::Weights *weights, int bnWeights) override;
  IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;
  bool isPlugin(const char *name) override;
  void destroyPlugin();

private:
  std::shared_ptr<PluginFactory> pluginFactory{new PluginFactory()};
  void (*nvPluginDeleter)(INvPlugin *){[](INvPlugin *ptr) { ptr->destroy(); }};

  //Normalize
  std::unordered_map<string, int> normalizeIDs = {
      std::make_pair("conv4_3_norm", 0),
      std::make_pair("conv5_3_norm", 1)};
  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> normalizeLayers[2]{{nullptr, nvPluginDeleter}, {nullptr, nvPluginDeleter}};

  //PriorBox
  float minSizes[4] = {32.0, 64.0, 128.0, 256.0};
  std::unordered_map<string, int> priorboxIDs = {
      std::make_pair("conv4_3_norm_mbox_priorbox", 0),
      std::make_pair("conv5_3_norm_mbox_priorbox", 1),
      std::make_pair("fc7_mbox_priorbox", 2),
      std::make_pair("conv6_2_mbox_priorbox", 3)};
  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> priorboxLayers[4]{
      nullPlugin, nullPlugin, nullPlugin, nullPlugin};

  
  std::shared_ptr<IPlugin> armConf{nullptr};
  std::shared_ptr<IPlugin> armLoc{nullptr};

  //DetectionOutput
  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> detection_out{nullptr, nvPluginDeleter};
};

} // namespace Shadow
#endif
