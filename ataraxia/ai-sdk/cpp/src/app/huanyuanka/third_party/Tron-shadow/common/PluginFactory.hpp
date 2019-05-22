#ifndef __PLUGINFACTORY_HPP___
#define __PLUGINFACTORY_HPP___

#include "Common.hpp"
#include "Util.hpp"
#include "SlicePlugin.hpp"
#include "RefineDetPlugin.hpp"
#include "DWConvPlugin.hpp"
#include "PreluPlugin.hpp"

namespace Shadow
{

class PluginFactory
{
  public:
	PluginFactory(){};

	/* 构建slice层参数：切分维度axis，切分点，输出数量。其中切分点与输出数二者选一，未设置切分点时，
	   将根据输出数在axis维度上平分输入tensor，因此需要输入在axis维度的长度可以整除。
	   当设置切分点时，输出数参数失效，函数只切分点切分tensor。
	   slice层目前不支持 N（BatchSize）通道的切分，axis取值范围 {-3,-2,-1,1,2,3}
	   slice_points 可以乱序，但是不能有重复 
	*/
	SlicePlugin *createSlicePlugin(int axis, vector<int> &slice_points, int n_slices = 0);
	SlicePlugin *createSlicePlugin(const void *serialData, size_t serialLength);

	/* 构建 RefineDet 模型的自定义层*/
	ApplyArmConf *createArmConfPlugin(const float);
	ApplyArmConf *createArmConfPlugin(const void *serialData, size_t serialLength);
	ApplyArmLoc *createArmLocPlugin();
	ApplyArmLoc *createArmLocPlugin(const void *serialData, size_t serialLength);

	/*  构建 DepthWise Convolution 层
	    使用时需要将卷机核的形状封装成KernelInfo格式，并将是否使用Bias传入。
	    struct KernelInfo{
		int kernelSize[2];
		int pad[2];
		int stride[2];};
		由于卷积核膨胀操作不常用，因此该操作作为一个公有成员函数单独使用，使用dalation(scale)调用方法，传入参数为膨胀系数
	*/
	DWConvPlugin *createDWConvPlugin(const Weights *weights, int nbWeights, int nbOutputChannels, KernelInfo kernelInfo, bool bias_term);
	DWConvPlugin *createDWConvPlugin(const void *serialData, size_t serialLength);
        
    /* 构建prelu层 */
    //PreluPlugin *createPreluPlugin(const Weights *weights, int nbWeights);
    PreluPlugin *createPreluPlugin(const char *weightsFile);
    PreluPlugin *createPreluPlugin(const Weights *weights, int nbWeights);
    PreluPlugin *createPreluPlugin(const void *serialData, size_t serialLength);
};

} // namespace Shadow
#endif
