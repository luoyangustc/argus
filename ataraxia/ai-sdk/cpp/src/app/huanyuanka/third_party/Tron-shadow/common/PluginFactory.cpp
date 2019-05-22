#include "PluginFactory.hpp"

namespace Shadow
{

SlicePlugin *PluginFactory::createSlicePlugin(int axis, vector<int> &slice_points, int n_slices)
{
	try
	{
		SlicePlugin *p = new SlicePlugin(axis, slice_points, n_slices);
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

SlicePlugin *PluginFactory::createSlicePlugin(const void *serialData, size_t serialLength)
{
	try
	{
		SlicePlugin *p = new SlicePlugin(serialData, serialLength);
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

ApplyArmConf *PluginFactory::createArmConfPlugin(const float objectness_score)
{
	try
	{
		ApplyArmConf *p = new ApplyArmConf(objectness_score);
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

ApplyArmConf *PluginFactory::createArmConfPlugin(const void *serialData, size_t serialLength)
{
	try
	{
		ApplyArmConf *p = new ApplyArmConf(serialData, serialLength);
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

ApplyArmLoc *PluginFactory::createArmLocPlugin()
{
	try
	{
		ApplyArmLoc *p = new ApplyArmLoc();
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

ApplyArmLoc *PluginFactory::createArmLocPlugin(const void *serialData, size_t serialLength)
{
	try
	{
		ApplyArmLoc *p = new ApplyArmLoc(serialData, serialLength);
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

DWConvPlugin *PluginFactory::createDWConvPlugin(const Weights *weights, int nbWeights, int nbOutputChannels, KernelInfo kernelInfo, bool bias_term)
{
	try
	{
		DWConvPlugin *p = new DWConvPlugin(weights, nbWeights, nbOutputChannels, kernelInfo, bias_term);
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

DWConvPlugin *PluginFactory::createDWConvPlugin(const void *serialData, size_t serialLength)
{
	try
	{
		DWConvPlugin *p = new DWConvPlugin(serialData, serialLength);
		return p;
	}
	catch (...)
	{
		return nullptr;
	}
}

PreluPlugin *PluginFactory::createPreluPlugin(const Weights *weights, int nbWeights){
    try
    {
        //PreluPlugin *p = new PreluPlugin(weights, nbWeights);
        PreluPlugin *p = new PreluPlugin(weights, nbWeights);
        return p;
    }
    catch (...)
    {
            return nullptr;
    }
}
PreluPlugin *PluginFactory::createPreluPlugin(const char *weightsFile)
{
        try
        {
                //PreluPlugin *p = new PreluPlugin(weights, nbWeights);
                PreluPlugin *p = new PreluPlugin(weightsFile);
                return p;
        }
        catch (...)
        {
                return nullptr;
        }
}

PreluPlugin *PluginFactory::createPreluPlugin(const void *serialData, size_t serialLength)
{
        try
        {
                PreluPlugin *p = new PreluPlugin(serialData, serialLength);
                return p;
        }
        catch (...)
        {
                return nullptr;
        }
}
} // namespace Shadow
