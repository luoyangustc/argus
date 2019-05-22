#include "RefineDetPlugin.hpp"

namespace Shadow
{
// ***************** ApplyArmLoc *******************************************
ApplyArmLoc::ApplyArmLoc(const void *buffer, size_t size)
{
    assert(size == sizeof(int));
    const int *d = reinterpret_cast<const int *>(buffer);
    _numPriorboxes = d[0];
}

Dims ApplyArmLoc::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(index == 0);
    assert(nbInputDims == 2);
    return inputs[1];
}

void ApplyArmLoc::configure(const Dims *inputDims, int nbInputs, const Dims *outputs, int nbOutputs, int batchSize)
{
    assert(nbInputs == 2);
    _numPriorboxes = inputDims[0].d[0] / 4;
}

int ApplyArmLoc::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workSpace, cudaStream_t stream)
{
    const float *arm_loc = reinterpret_cast<const float *>(inputs[0]);
    const float *priorbox_loc = reinterpret_cast<const float *>(inputs[1]);
    float *decodePriorbox = reinterpret_cast<float *>(outputs[0]);
    applyLoc(batchSize, _numPriorboxes, arm_loc, priorbox_loc, decodePriorbox);
    return 0;
}

size_t ApplyArmLoc::getSerializationSize()
{
    return sizeof(int);
}

void ApplyArmLoc::serialize(void *buffer)
{
    int *d = reinterpret_cast<int *>(buffer);
    d[0] = _numPriorboxes;
}

// ******** ApplyArmConf ******************************************************
ApplyArmConf::ApplyArmConf(const void *buffer, size_t size)
{
    assert(size == 3 * sizeof(float));
    const float *d = reinterpret_cast<const float *>(buffer);
    _objectness_score = d[0];
    _numClasses = d[1];
    _numPriorboxes = d[2];
}

Dims ApplyArmConf::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(index == 0);
    assert(nbInputDims == 2);
    return inputs[1];
}

void ApplyArmConf::configure(const Dims *inputDims, int nbInputs, const Dims *outputs, int nbOutputs, int batchSize)
{
    assert(nbInputs == 2);
    _numPriorboxes = static_cast<float>(inputDims[0].d[0] / 2);
    _numClasses = static_cast<float>(inputDims[1].d[0] / _numPriorboxes);
}

int ApplyArmConf::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workSpace, cudaStream_t stream)
{
    const float *arm_conf = reinterpret_cast<const float *>(inputs[0]);
    const float *odm_conf = reinterpret_cast<const float *>(inputs[1]);
    float *conf = reinterpret_cast<float *>(outputs[0]);
    applyConf(batchSize, _numPriorboxes, _numClasses, _objectness_score, arm_conf, odm_conf, conf, stream);
    return 0;
}

size_t ApplyArmConf::getSerializationSize()
{
    return 3 * sizeof(float);
}

void ApplyArmConf::serialize(void *buffer)
{
    float *d = reinterpret_cast<float *>(buffer);
    d[0] = _objectness_score;
    d[1] = _numClasses;
    d[2] = _numPriorboxes;
}

} // namespace Shadow
