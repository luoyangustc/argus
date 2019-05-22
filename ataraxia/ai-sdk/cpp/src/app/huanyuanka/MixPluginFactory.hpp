#pragma once

#include <assert.h>
#include <string.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "PluginFactory.hpp"

namespace tron {
namespace mix {

class MixPluginFactory : public nvinfer1::IPluginFactory,
                         public nvcaffeparser1::IPluginFactory {
 public:
  using INvPlugin = nvinfer1::plugin::INvPlugin;

  // 继承自nvcaffeparser1::IPluginFactory
  nvinfer1::IPlugin *
  createPlugin(const char *layerName,
               const nvinfer1::Weights *weights,
               int nbWeights) override {
    if (!strcmp(layerName, "detection_out")) {
      detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
          createSSDDetectionOutputPlugin(
              {true,
               false,
               0,
               8,
               500,
               400,
               0.0500000007451,
               0.300000011921,
               nvinfer1::plugin::CodeTypeSSD::CENTER_SIZE,
               {0, 1, 2},
               false,
               true}),
          nvPluginDeleter);
      return detection_out.get();
    } else if (priorboxIDs.find(std::string(layerName)) != priorboxIDs.end()) {
      const int i = priorboxIDs[layerName];
      switch (i) {
        case 0: {
          float minSize = 20.0, maxSize = 48.0, aspectRatio[] = {1.0, 2.0};
          priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
              createSSDPriorBoxPlugin(
                  {&minSize,
                   &maxSize,
                   aspectRatio,
                   1,
                   1,
                   2,
                   true,
                   false,
                   {0.10000000149, 0.10000000149, 0.20000000298, 0.20000000298},
                   0,
                   0,
                   8.0,
                   8.0,
                   0.5}),
              nvPluginDeleter);
          break;
        }
        case 1: {
          float minSize = 48.0, maxSize = 96.0, aspectRatio[] = {1.0, 2.0, 3.0};
          priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
              createSSDPriorBoxPlugin(
                  {&minSize,
                   &maxSize,
                   aspectRatio,
                   1,
                   1,
                   3,
                   true,
                   false,
                   {0.10000000149, 0.10000000149, 0.20000000298, 0.20000000298},
                   0,
                   0,
                   16.0,
                   16.0,
                   0.5}),
              nvPluginDeleter);
          break;
        }

        case 2: {
          float minSize = 96.0,
                maxSize = 132.0,
                aspectRatio[] = {1.0, 2.0, 3.0};
          priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
              createSSDPriorBoxPlugin(
                  {&minSize,
                   &maxSize,
                   aspectRatio,
                   1,
                   1,
                   3,
                   true,
                   false,
                   {0.10000000149, 0.10000000149, 0.20000000298, 0.20000000298},
                   0,
                   0,
                   32.0,
                   32.0,
                   0.5}),
              nvPluginDeleter);
          break;
        }

        case 3: {
          float minSize = 132.0,
                maxSize = 172.0,
                aspectRatio[] = {1.0, 2.0, 3.0};
          priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
              createSSDPriorBoxPlugin(
                  {&minSize,
                   &maxSize,
                   aspectRatio,
                   1,
                   1,
                   3,
                   true,
                   false,
                   {0.10000000149, 0.10000000149, 0.20000000298, 0.20000000298},
                   0,
                   0,
                   64.0,
                   64.0,
                   0.5}),
              nvPluginDeleter);
          break;
        }

        case 4: {
          float minSize = 164.0, maxSize = 208.0, aspectRatio[] = {1.0, 2.0};
          priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
              createSSDPriorBoxPlugin(
                  {&minSize,
                   &maxSize,
                   aspectRatio,
                   1,
                   1,
                   2,
                   true,
                   false,
                   {0.10000000149, 0.10000000149, 0.20000000298, 0.20000000298},
                   0,
                   0,
                   100.0,
                   100.0,
                   0.5}),
              nvPluginDeleter);
          break;
        }
      }
      return priorboxLayers[i].get();
    } else if (pReluIDs.find(std::string(layerName)) != pReluIDs.end()) {
      // std::cout<<layerName<<std::endl;
      const int i = pReluIDs[layerName];
      // const char *weightsFile = weightsName[i];
      pReluLayers[i] = std::shared_ptr<nvinfer1::IPlugin>(
          pluginFactory.get()->createPreluPlugin(weights, nbWeights));
      return pReluLayers[i].get();
    } else {
      std::cout << layerName << std::endl;
      assert(0);
      return nullptr;
    }
  };

  // 继承自nvinfer1::IPluginFactory
  nvinfer1::IPlugin *createPlugin(const char *layerName,
                                  const void *serialData,
                                  size_t serialLength) override {
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "detection_out")) {
      detection_out = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
          nvinfer1::plugin::createSSDDetectionOutputPlugin(serialData,
                                                           serialLength),
          nvPluginDeleter);
      return detection_out.get();
    } else if (priorboxIDs.find(std::string(layerName)) != priorboxIDs.end()) {
      const int i = priorboxIDs[layerName];
      priorboxLayers[i] = std::unique_ptr<INvPlugin, void (*)(INvPlugin *)>(
          createSSDPriorBoxPlugin(serialData, serialLength),
          nvPluginDeleter);
      return priorboxLayers[i].get();
    } else if (pReluIDs.find(std::string(layerName)) != pReluIDs.end()) {
      const int i = pReluIDs[layerName];
      pReluLayers[i] = std::shared_ptr<nvinfer1::IPlugin>(
          pluginFactory.get()->createPreluPlugin(serialData, serialLength));
      return pReluLayers[i].get();
    } else {
      std::cout << layerName << std::endl;
      assert(0);
      return nullptr;
    }
  }

  bool isPlugin(const char *name) override {
    return (!strcmp(name, "detection_out") ||
            priorboxIDs.find(std::string(name)) != priorboxIDs.end() ||
            pReluIDs.find(std::string(name)) != pReluIDs.end());
  }

  void destroyPlugin() {
    for (unsigned i = 0; i < priorboxIDs.size(); ++i) {
      priorboxLayers[i].reset();
    }
    for (unsigned i = 0; i < pReluIDs.size(); i++) {
      pReluLayers[i].reset();
    }
    detection_out.reset();
  }

  void (*nvPluginDeleter)(INvPlugin *){
      [](INvPlugin *ptr) { ptr->destroy(); }};
  // DetectionOutput
  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> detection_out{
      nullptr,
      nvPluginDeleter};
  // PriorBox
  std::unordered_map<std::string, int> priorboxIDs = {
      std::make_pair("conv4_3_norm_mbox_priorbox", 0),
      std::make_pair("fc7_mbox_priorbox", 1),
      std::make_pair("conv6_2_mbox_priorbox", 2),
      std::make_pair("conv7_2_mbox_priorbox", 3),
      std::make_pair("conv8_2_mbox_priorbox", 4)};
  std::unique_ptr<INvPlugin, void (*)(INvPlugin *)> priorboxLayers[5]{
      {nullptr, nvPluginDeleter},
      {nullptr, nvPluginDeleter},
      {nullptr, nvPluginDeleter},
      {nullptr, nvPluginDeleter},
      {nullptr, nvPluginDeleter}};

  std::unordered_map<string, int> pReluIDs = {
      std::make_pair("prelu1", 0),
      std::make_pair("prelu2", 1),
      std::make_pair("prelu3", 2),
      std::make_pair("prelu4", 3),
      std::make_pair("prelu5", 4),
  };
  const char *weightsName[5] = {
      "data/prelu/weights1.txt",
      "data/prelu/weights2.txt",
      "data/prelu/weights3.txt",
      "data/prelu/weights4.txt",
      "data/prelu/weights5.txt"};
  std::shared_ptr<nvinfer1::IPlugin> pReluLayers[5]{nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr};
  std::shared_ptr<Shadow::PluginFactory> pluginFactory{
      new Shadow::PluginFactory()};
};

}  // namespace mix
}  // namespace tron
