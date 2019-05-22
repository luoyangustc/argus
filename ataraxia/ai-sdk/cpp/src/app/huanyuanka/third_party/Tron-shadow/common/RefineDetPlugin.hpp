#ifndef __REFINEDETPLUGIN_HPP___
#define __REFINEDETPLUGIN_HPP___

#include "Common.hpp"
#include "Util.hpp"

namespace Shadow
{

void applyConf(int batchSize, int _numPriorboxes, int _numClasses, float _objectness_score, const float *arm_conf, const float *odm_conf, float *conf, cudaStream_t stream);

void applyLoc(int batchSize, int _numPriorboxes, const float *arm_loc, const float *priorbox_loc, float *loc);

class ApplyArmLoc : public IPlugin
{
public:
  ApplyArmLoc(){};
  ApplyArmLoc(const void *buffer, size_t size);

  inline int getNbOutputs() const override { return 1; };
  Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;
  void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int maxBatchSize) override;

  int initialize() override { return 0; };
  void terminate() override{};
  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; };
  int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workSpace, cudaStream_t stream) override;

  size_t getSerializationSize() override;
  void serialize(void *buffer) override;

protected:
  int _numPriorboxes;
};

class ApplyArmConf : public IPlugin
{
public:
  ApplyArmConf(const float objectness_score) : _objectness_score(objectness_score){};
  ApplyArmConf(const void *buffer, size_t size);

  inline int getNbOutputs() const override { return 1; }
  Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;
  void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int maxBatchSize) override;

  int initialize() override { return 0; }
  void terminate() override{};
  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; };
  int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workSpace, cudaStream_t stream) override;

  size_t getSerializationSize() override;
  void serialize(void *buffer) override;

protected:
  float _objectness_score;
  float _numClasses;
  float _numPriorboxes;
};

} // namespace Shadow
#endif
