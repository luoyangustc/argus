#ifndef __VSEPAPLUGINFACTORY_HPP___
#define __VSEPAPLUGINFACTORY_HPP___

#include "PluginFactory.hpp"

namespace Shadow
{
class VsepaPluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
  virtual IPlugin *createPlugin(const char *layerName, const nvinfer1::Weights *weights, int bnWeights) override;
  IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;
  bool isPlugin(const char *name) override
  {
    return isPluginExt(name);
  }
  bool isPluginExt(const char *layerName) override;
  void destroyPlugin();

private:
  std::shared_ptr<PluginFactory> pluginFactory{new PluginFactory()};
  std::shared_ptr<IPlugin> slice{nullptr};
};

} // namespace Shadow
#endif
