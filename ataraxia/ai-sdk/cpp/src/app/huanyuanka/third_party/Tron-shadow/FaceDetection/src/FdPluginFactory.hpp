#ifndef __MOBILEPLUGINFACTORY_HPP___
#define __MOBILEPLUGINFACTORY_HPP___

#include "PluginFactory.hpp"

#define nullPlugin           \
  {                          \
    nullptr, nvPluginDeleter \
  }

namespace Shadow
{
class FdPluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
  FdPluginFactory(){}
  virtual IPlugin *createPlugin(const char *layerName, const nvinfer1::Weights *weights, int bnWeights) override;
  IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;
  bool isPlugin(const char *name) override;
  void destroyPlugin();

private:
  std::shared_ptr<PluginFactory> pluginFactory{new PluginFactory()};
  void (*nvPluginDeleter)(INvPlugin *){[](INvPlugin *ptr) { ptr->destroy(); }};

  //Normalize
  std::unordered_map<string, int> normalizeIDs = {
      std::make_pair("block_5_2_norm", 0),
      std::make_pair("block_4_6_norm", 1)};
  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> normalizeLayers[2]{{nullptr, nvPluginDeleter}, {nullptr, nvPluginDeleter}};

  //PriorBox
  float minSizes[5] = {16.0, 76.0, 136.0, 196.0, 256.0};
  std::unordered_map<string, int> priorboxIDs = {
      std::make_pair("block_4_6_norm_mbox_priorbox", 0),
      std::make_pair("block_5_2_norm_mbox_priorbox", 1),
      std::make_pair("block_6_2_mbox_priorbox", 2),
      std::make_pair("block_7_1_mbox_priorbox", 3),
      std::make_pair("block_8_1_mbox_priorbox", 4)};

  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> priorboxLayers[5]{
      nullPlugin, nullPlugin, nullPlugin, nullPlugin, nullPlugin};

  
  std::shared_ptr<IPlugin> armConf{nullptr};
  std::shared_ptr<IPlugin> armLoc{nullptr};

  
  std::unordered_map<string, int> pReluIDs = {
    std::make_pair("prelu1", 0),
    std::make_pair("prelu2", 1),
    std::make_pair("prelu3", 2),
    std::make_pair("prelu4", 3),
    std::make_pair("prelu5", 4),
  };
  const char *weightsName[5] = {"data/prelu/weights1.txt", "data/prelu/weights2.txt", "data/prelu/weights3.txt", "data/prelu/weights4.txt", "data/prelu/weights5.txt"};
  std::shared_ptr<IPlugin> pReluLayers[5]{nullptr, nullptr, nullptr, nullptr, nullptr};

  //DetectionOutput
  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> detection_out{nullptr, nvPluginDeleter};
};

} // namespace Shadow
#endif
