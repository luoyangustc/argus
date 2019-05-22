#ifndef __RESIZEBIPLUGINFACTORY_H___
#define __RESIZEBIPLUGINFACTORY_H___

#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <memory>
#include <string.h>
#include <string>
#include <vector>
#include <unordered_map>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"
#include "ResizeBiPlugin.hpp"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;
using namespace std;

class ResizeBiPluginFactory: public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory
{
public:
    
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const FieldCollection fc) override
    {
        assert(isPlugin(layerName));
        const nvuffparser::FieldMap* fields = fc.fields;
        int nbFields = fc.nbFields;
        std::cout<<layerName<<std::endl;
        if(strcmp(layerName,"_ResizeBilinear")==0)
        {   
            i++;
            my_height = resizeBiIDs[i][0];
            my_weight = resizeBiIDs[i][1];
           // cout<<my_height<<"..."<<my_weight<<endl;
            resizeBiLayers[i] = std::unique_ptr<ResizeBiPlugin>(new ResizeBiPlugin(my_height, my_weight));

            return resizeBiLayers[i].get();
        }
        
        else
       {
            //std::cout << layerName << std::endl;
            assert(0);
            return nullptr;
        }
    };
    //继承自nvinfer1::IPluginFactory
    IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));
        if (strcmp(layerName, "_ResizeBilinear")==0)
        {
            assert(resizeBiLayers == nullptr);
            k++;
            resizeBiLayers[k] = std::unique_ptr<ResizeBiPlugin>(new ResizeBiPlugin(serialData, serialLength));
            return resizeBiLayers[k].get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
        
 }
    
    bool isPlugin(const char *name) override
    {
		cout<<name<<endl;
        return (strcmp(name, "_ResizeBilinear")==0);
    }
    
    void destroyPlugin()
    {
        for(int i;i<7;i++)
			resizeBiLayers[i].reset();
    }
    
    
    //resize_bilinear
    int my_height, my_weight;
    int i = -1;
    int k= -1;
	//parameters
    vector<vector<int>> resizeBiIDs=
    {
        {16,12},
        {32,24},
        {64,48},
        {64,48},
        {64,48},
        {64,48},
        {64,48}
    };
    std::unique_ptr<ResizeBiPlugin> resizeBiLayers[7]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
};



#endif
