#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <cstdlib>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <regex>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"
#include "NvUtils.h"
#include "common.h"
#include "ResizeBiPluginFactory.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace nvuffparser;
using namespace nvinfer1;
using namespace cv;
using namespace std;
using namespace plugin;
static Logger gLogger;
const char* INPUT_BLOB_NAME = "tower_0/Placeholder";
const char* OUTPUT_BLOB_NAME = "tower_0/refine_out/BatchNorm/FusedBatchNorm";
const char* UFF_MODEL_PATH = "cpn_freeze.pb.uff";
char engin_file_name[] = "cpn_engin.bin";
const string Image_path = "images";
float pre_param[4] = {102.9801, 115.9465, 122.7717, 0};
const int gpuID = 0;
const int batch_size = 1;
const int iteration = 1;
const int input_shape[3] = {3, 256, 192};
const string suffix = ".*.jpg";
const int BATCH_SIZE = 1;
const int INPUT_WIDTH = 192;
const int INPUT_HEIGHT = 256;
const int INPUT_CHANNELS= 3;
ICudaEngine *engine;
IExecutionContext *context;
IUffParser* parser;
IRuntime* runtime;
void *buffers[2];
int input_index = 0;
int output_index = 0;

void doInference(float* input_data, float* output_data, int batch_size);
void predict(const vector<Mat> &imgs, const vector<string> &attributes, vector<string> &results);
void init(const int gpuID, void *data, const int size);
void tfToTRTModel(const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME, const char* UFF_MODEL_PATH, IHostMemory *&trt_model_stream, unsigned int max_batch_size, nvuffparser::IPluginFactory* pluginFactory);

#define RETURN_AND_LOG(ret, severity, message)                                              \
do {                                                                                    \
std::string error_message = "sample_uff_cpn: " + std::string(message);            \
gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
return (ret);                                                                       \
} while(0)

inline vector<string> mySplit(string my_str,string seperate)
{
    vector<string> result;
    size_t split_index = my_str.find(seperate);
    size_t start = 0;
    
    while(string::npos!=split_index)
    {
        result.push_back(my_str.substr(start,split_index-start));
        start = split_index+seperate.size();
        split_index = my_str.find(seperate,start);
    }
    result.push_back(my_str.substr(start,split_index-start));
    return result;
}

inline void getAllFiles(string path, string suffix, vector<string> &files)
{
    regex reg_obj(suffix, regex::icase);
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(path.c_str())) == NULL)
    {
        return ;
    }
    else
    {
        while((dirp = readdir(dp)) != NULL)
        {
            if(dirp->d_type == 8 && regex_match(dirp->d_name, reg_obj))
            {
                string file_absolute_path = path.c_str();
                file_absolute_path = file_absolute_path.append("/");
                file_absolute_path = file_absolute_path.append(dirp->d_name);
                files.push_back(file_absolute_path);
            }
        }
    }
    closedir(dp);
}


void tfToTRTModel(const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME, const char* UFF_MODEL_PATH, IHostMemory *&trt_model_stream, unsigned int max_batch_size, nvuffparser::IPluginFactory* pluginFactory)
{
    Logger g_logger;
    IUffParser* parser = createUffParser();
    try
    {
        parser->registerInput(INPUT_BLOB_NAME, Dims3(input_shape[0], input_shape[1], input_shape[2]), UffInputOrder::kNCHW);
        parser->registerOutput(OUTPUT_BLOB_NAME);
    }
    catch(...)
    {
        cerr << "Parse uff file failed..." << endl;
    }
    IBuilder* builder = createInferBuilder(g_logger);
    INetworkDefinition* network = builder->createNetwork();
	parser->setPluginFactory(pluginFactory);
	//cout << "parse 11\n";
    if(!parser->parse(UFF_MODEL_PATH, *network, nvinfer1::DataType::kFLOAT))
    {
        cerr<<"fail to parse uff file"<<endl;
        exit(1);
    }
	//cout << "parse 22\n";
    builder->setMaxBatchSize(max_batch_size);
    builder->setMaxWorkspaceSize(256 << 20);
    ICudaEngine *engine;
    try
    {
        engine = builder->buildCudaEngine(*network);
    }
    catch (...)
    {
        cerr << "Build cuda engine failed, please cheack blob name setted" << endl;
    }
    
    network->destroy();
    parser->destroy();
	//cout<<"-----3";
    trt_model_stream = engine->serialize();
	
    engine->destroy();
    builder->destroy();
}

void init(const int gpuID, void *data, const int size)
{
	ResizeBiPluginFactory pluginFactory;
    try
    {
        cudaSetDevice(gpuID);
    }
    catch (...)
    {
        return ;
    }
    
    try
    {
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(data, size, &pluginFactory);
        context = engine->createExecutionContext();
    }
    catch(...)
    {
        return;
    }
    try
    {
        input_index = engine->getBindingIndex(INPUT_BLOB_NAME);
        output_index = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    }
    catch(...)
    {
        return ;
    }
    try
    {
        cudaMalloc(&buffers[input_index], BATCH_SIZE * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float));
        cudaMalloc(&buffers[output_index], 4*BATCH_SIZE * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float));
    }
    catch (...)
    {
        return;
    }
    
}
void predict(const vector<Mat> &imgs, const vector<string> &attributes, vector<float> &results)
{
    vector<float> data;
	vector<float> network_out;
    Mat img, img_tmp,img_ori;
    for(int i = 0;i < BATCH_SIZE; ++i)
    {
        img_ori = imgs[i];
		resize(img_ori,img_tmp,Size(INPUT_WIDTH,INPUT_HEIGHT));
        for(int row=0;row<INPUT_HEIGHT;row++)
        {
            for(int col = 0;col<INPUT_WIDTH;col++)
            {
                img.at<Vec3d>(row,col)[0] = img_tmp.at<Vec3d>(row,col)[0] - pre_param[2];
                img.at<Vec3d>(row,col)[1] = img_tmp.at<Vec3d>(row,col)[1] - pre_param[1];
                img.at<Vec3d>(row,col)[2] = img_tmp.at<Vec3d>(row,col)[2] - pre_param[0];
            }
        }
        img = img/255.;
        for(int c=0; c < INPUT_CHANNELS; ++c)
        {
            for(int row=0; row < INPUT_HEIGHT; row++)
            {
                for(int col=0; col < INPUT_WIDTH; col++)
                {
                    data.push_back(img.at<Vec3f>(row,col)[c]);
                }
            }
        }
        doInference(&data[0], &network_out[0], BATCH_SIZE);
        float *outdata = NULL;
        outdata = &network_out[0];
        for(int i=0;i<4;i++)
        {
            results.push_back(outdata[i]);
        }
    }
}

void doInference(float* input_data, float* output_data, int batch_size)
{
    float * tmp_data = NULL;
    
    for (int i = 0; i < 1; i++)
    {
        float total = 0;
        for (int run = 0; run < 1; run++)
        {
            //*create space for input and set the input data*/
            tmp_data = &input_data[0] + run * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
            try
            {
                cudaMemcpyAsync(buffers[input_index], tmp_data, 1 * INPUT_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
            }
            catch (...)
            {
                return ;
            }
            
            context->execute(batch_size, &buffers[0]);
            /*
             auto t_start = chrono::high_resolution_clock::now();
             auto t_end = chrono::high_resolution_clock::now();
             ms = chrono::duration<float, milli>(t_end - t_start).count();
             total += ms;
             total /= batch_size;
             cout << "Average over " << run_num << " runs is " << total << " ms." << endl;
             */
            tmp_data = &output_data[0] + 4 * sizeof(float);
            try
            {
                cudaMemcpyAsync(tmp_data, buffers[output_index], 4 * sizeof(float), cudaMemcpyDeviceToHost);
            }
            catch (...)
            {
                return ;
            }
        }
    }
    tmp_data = NULL;
}


int main(int argc, char** argv)
{
    int engin_size = 0;
	ResizeBiPluginFactory pluginFactory;
    if (argc == 2 && atoi(argv[1]))
    {
        IHostMemory* trt_model_stream{nullptr};
        cout << "********serialize begin*******" << endl;
        tfToTRTModel(INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, UFF_MODEL_PATH, trt_model_stream, batch_size, &pluginFactory);
		//pluginFactory.destroyPlugin();
        FILE *file = fopen(engin_file_name, "wb+");
        if (!file)
        {
            cerr << "Can't open file to write TRT model" << endl;
            exit(-1);
        }
        engin_size = trt_model_stream->size();
        fwrite(trt_model_stream->data(), 1, engin_size, file);
        fclose(file);
        trt_model_stream->destroy();
    }
    
    auto *model_fp = fopen(engin_file_name, "rb");
    if (!model_fp)
    {
        cerr << "Can not open engine file" << endl;
        exit(-1);
    }
    fseek(model_fp, 0, SEEK_END);
    engin_size = ftell(model_fp);
    rewind(model_fp);
    cout<<engin_size<<endl;
    void *data = malloc(engin_size);
    if (!data)
    {
        cerr << "Alloc engine model memory error" << endl;
        exit(-1);
    }
    fread(data, 1, engin_size, model_fp);
    fclose(model_fp);
    
    vector<string> files;
    vector<string> split_result;
    string tmp_name;
    
    getAllFiles(Image_path, suffix, files);
    
    sort(files.begin(), files.end());
    init(gpuID, data, engin_size);
    
    if(files.size() == 0)
    {
        cerr<<"Invalid data"<<endl;
        return -1;
    }
    
    free(data);
    int rounds = files.size() / batch_size;
    
    for(int i = 0; i < rounds; i++)
    {
        vector<Mat> imgs;
        vector<string> img_name;
        vector<string> attributes;
        vector<float> results;
        
        for(int j = 0; j < batch_size; j++)
        {
            Mat img = cv::imread(files[i * batch_size + j]);
            split_result = mySplit(files[i * batch_size + j], "/");
            tmp_name = split_result[split_result.size() - 1];
            img_name.push_back(tmp_name);
            

            if (!img.data)
            {
                cerr << "Read image " << files[i * batch_size + j] <<" error, No Data!" << endl;
                continue;
            }
            imgs.push_back(img);
        }
        
        predict(imgs, attributes, results);
        vector<float>::iterator iter;
        for(iter=results.begin();iter!=results.end();iter++)
        {
            cout<<*iter<<",";
        }
        cout<<endl;
    }
    
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaFree(buffers[3]);
    
    context->destroy();
    engine->destroy();
    runtime->destroy();
    

    return 0;
}
