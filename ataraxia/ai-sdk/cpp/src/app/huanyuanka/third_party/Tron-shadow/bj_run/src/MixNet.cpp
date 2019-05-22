#include "MathFunction.hpp"
#include "MixNet.hpp"
#include "face_alignment.hpp"
#include <time.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace rapidjson;

namespace Shadow
{

static Logger gLogger;
static Profiler gProfiler;

inline size_t getDimSize(DimsCHW &dim)
{
    return dim.c() * dim.h() * dim.w();
}

//在nums[begin:end]的数组中，找到最大值的索引
int GetIndexOfMaxNumber(float *nums, int begin, int end)
{
    int ans = begin;
    float max = nums[begin];
    for (int i = begin; i < end; i++)
        if (nums[i] > max)
        {
            max = nums[i];
            ans = i;
        }
    return ans - begin;
}

MixNet::MixNet(int modelNum_) 
{   
    models.resize(modelNum_);
    for ( size_t i = 0; i<modelNum_; i++)
    {
        models[i].engine = nullptr;
        models[i].context = nullptr;
        models[i].maxBatchsize = 0;
    }
    for (int i = 0; i < 48; i++)
    {
        if (i < 7 || (i >= 9 && i < 11) || (i >= 12 && i < 19))
            bkLabel[i] = "terror";
        else if ((i >= 7 && i < 9) || (i >= 42 && i < 44))
            bkLabel[i] = "march";
        else if (i == 11 || (i >= 28 && i < 30))
            bkLabel[i] = "text";
        else
            bkLabel[i] = "normal";
    }
}

ShadowStatus MixNet::init(const int gpuID, const std::vector<std::vector<char>> data,const std::vector<int> size)
{
    if (data.size() == 0 || size.size() == 0)
        return shadow_status_invalid_gie_file;
    try
    {
        cudaSetDevice(gpuID);
    }
    catch (...)
    {
        return shadow_status_set_gpu_error;
    }
    try
    {
        infer = createInferRuntime(gLogger);
        for(size_t i = 0; i < models.size(); i++)
        {
            models[i].engine  = infer->deserializeCudaEngine((void*)(data.at(i).data()), size.at(i), &pluginFactory);
            models[i].context = models[i].engine->createExecutionContext();
            models[i].context->setProfiler(&(models[i].gProfiler));
            models[i].maxBatchsize = models[i].engine->getMaxBatchSize();
        }
    }
    catch (...)
    {
        return shadow_status_deserialize_error;
    }
    try
    {
        cudaStreamCreate(&stream);
    }
    catch (...)
    {
        shadow_status_create_stream_error;
    }
    return allocateMemory();
}

//初始化网络 will be delete
ShadowStatus MixNet::init(const int gpuID, void *data, int size)
{
    return shadow_status_success;
}

ShadowStatus MixNet::allocateMemory()
{   

    for(size_t model_id = 0; model_id < models.size(); model_id++)
    {
        Model &model = models[model_id];
        int bindings = model.engine->getNbBindings();
        ////printf("bindings %d\n", n);
        for (int i = 0; i < bindings; i++)
        {
            Info info;
            info.index = i;
            info.dim = static_cast<DimsCHW &&>(model.engine->getBindingDimensions(i));
            info.dataNum = getDimSize(info.dim);
            size_t memory = model.maxBatchsize * info.dataNum * sizeof(float);//using fd index 0 ,malloc bathsize memory for detectionputengin input
            try
            {
                info.data = (float *)malloc(memory);
                assert(info.data);
            }
            catch (...)
            {
                return shadow_status_host_malloc_error;
            }
            try
            {
                cudaMalloc(&info.data_gpu, memory);
                assert(info.data_gpu);
            }
            catch (...)
            {
                return shadow_status_cuda_malloc_error;
            }
            
            if (model.engine->bindingIsInput(i))
            {
                model.inputInfo.push_back(info);
            }
            else
            {
                model.outputInfo.push_back(info);
            }
        }
    }
    return shadow_status_success;
}


void MixNet::processImageGPU(const vector<cv::Mat> &imgs, int model_id, InterMethod method)
{
    void *data_gpu = models[model_id].inputInfo[0].data_gpu;
    DimsCHW dim = models[model_id].inputInfo[0].dim;
    size_t dataNum = models[model_id].inputInfo[0].dataNum;
    for (size_t b = 0; b < imgs.size(); b++)
    {
        //printf("%lu\n",b);
        cv::Mat img = imgs[b];
        inputGpuMat.upload(img);
        process(inputGpuMat, (float *)(data_gpu) + b * dataNum,
                dim.w(), dim.h(), preParams[model_id], method);
    }
}

void **MixNet::getBuffers(int model_id){

    int n = models[model_id].engine->getNbBindings();
    vector<Info> &input = models[model_id].inputInfo;
    vector<Info> &output = models[model_id].outputInfo;
    void **buffers = (void**)malloc(sizeof(void*) * n);
    for(int i = 0; i < input.size(); i++){
        buffers[input[i].index] = input[i].data_gpu;
    }
    for(int i = 0; i < output.size(); i++){
        buffers[output[i].index] = output[i].data_gpu;
    }
    return buffers;
}

void MixNet::cropFace(vector<cv::Mat> imgs, vector<FaceBox> &faceBoxes, vector<cv::Mat> &faces){
    float scale = 0.0143;
    for(int i = 0; i < faceBoxes.size(); i++){
        FaceBox &faceBox = faceBoxes[i];
        int imageId =faceBox.imageId;
        int width = faceBox.xmax - faceBox.xmin + 1;
        int height = faceBox.ymax - faceBox.ymin + 1;

        faceBox.xmin = std::max(0, cvRound(faceBox.xmin - scale * width));
        faceBox.ymin = std::max(0, cvRound(faceBox.ymin - scale * height));
        faceBox.xmax = std::min(originSize[imageId].first  - 1, cvRound(faceBox.xmax + scale * width));
        faceBox.ymax = std::min(originSize[imageId].second - 1, cvRound(faceBox.ymax + scale * height));

        width = faceBox.xmax - faceBox.xmin + 1;
        height = faceBox.ymax - faceBox.ymin + 1;

        cv::Rect rect(faceBox.xmin, faceBox.ymin, width, height);
        cv::Mat face(imgs[faceBox.imageId], rect);
        transpose(face,face);
        faces.push_back(face);
    }
}


ShadowStatus MixNet::predict(const vector<cv::Mat> &imgs, const vector<string> &attributes, vector<string> &json_results){
    originSize.clear();
    for(int i = 0; i < imgs.size(); i++){
        originSize.push_back(make_pair(imgs[i].cols, imgs[i].rows));
    }
    
    if (imgs.size() == 0)
    {
        return shadow_status_batchsize_zero_error;
    }
    if (imgs.size() > models[0].maxBatchsize)
    {
        cout << "Please input the valid number of images: max = " << models[0].maxBatchsize << endl;
        return shadow_status_batchsize_exceed_error;
    }
    //std::cout<<"shadow predict: "<<imgs.size()<<std::endl;
    //模型一的结果
    vector<vector<int>> mixResults;
    //模型一输出的人脸框
    vector<FaceBox> boxes;
    //模型二过滤后剩下的人脸id(boxes中模型二conf>0.7的人脸下标)
    vector<int> faceIds;
    //模型二的输入图片
    vector<cv::Mat> faces;
    //模型三的输入图片
    vector<cv::Mat> alignedFaces;

    predicModel(imgs, 0);
    dealMixResult(boxes, mixResults, imgs.size());

    cropFace(imgs, boxes, faces);
    int round = faces.size() / models[1].maxBatchsize;
    for(int i = 0; i <= round; i++){
        vector<FaceBox*> boxBatch;
        vector<cv::Mat> faceBatch;
        size_t batch = models[1].maxBatchsize;
        size_t size = (i < round ? batch : faces.size() % batch);
        if(size == 0)
            continue;
        for(int j = 0; j < size; j++){
            boxBatch.push_back(&boxes[i * batch + j]);
            faceBatch.push_back(faces[i * batch + j]);
        }
        predicModel(faceBatch, 1);
        dealDet3Result(boxBatch, faceBatch.size());
    }

    for(int i = 0; i < boxes.size(); i++){
        if(boxes[i].conf >= 0.7){
            faceIds.push_back(i);
            faceAlignmet(imgs[boxes[i].imageId], boxes[i].points, alignedFaces);
        }
    }

    round = alignedFaces.size() / models[2].maxBatchsize;
    for(int i = 0; i <= round; i++){
        vector<FaceBox*> boxBatch;
        vector<cv::Mat> faceBatch;
        size_t batch = models[2].maxBatchsize;
        size_t size = (i < round ? models[2].maxBatchsize : alignedFaces.size() % models[2].maxBatchsize);
        if(size == 0)
            continue;
        for(int j = 0; j < size; j++){
            int faceId = faceIds[i * models[2].maxBatchsize + j];
            boxBatch.push_back(&boxes[faceId]);
            faceBatch.push_back(alignedFaces[i * models[2].maxBatchsize + j]);
        }
        predicModel(faceBatch, 2, InterMethod::nearest);
        saveFeature(boxBatch);
    }


    int faceId = 0;
    for(int i = 0;i < imgs.size();i++){
        vector<FaceBox*> faces;
        while(faceId < boxes.size() && boxes[faceId].imageId == i)
            faces.push_back(&boxes[faceId++]);
        json_results.push_back(getResultJson(mixResults[i], faces));
    }

    // save imgs with face rect and points
    /*
    for(int i = 0; i < boxes.size(); i++){
        FaceBox &face = boxes[i];
        // if(face.conf < 0.7)
        //     continue;
        cv::Mat &img = imgs[face.imageId];
        if(face.conf < 0.7){
            cv::rectangle(img, cv::Point(face.xmin,face.ymin), cv::Point(face.xmax, face.ymax), {0,0,255});
            for(int i = 0; i < 5; i++)
                cv::circle(img, face.points[i], 1, {0,0,255});
        }
        else{
            cv::rectangle(img, cv::Point(face.xmin,face.ymin), cv::Point(face.xmax, face.ymax), {0,255,0});
            for(int i = 0; i < 5; i++)
                cv::circle(img, face.points[i], 1, {0,255,0});
        }

    }
    for(int i = 0; i < imgs.size(); i++)
        cv::imwrite("results/" + to_string(name++) + ".jpg",imgs[i]);
    */
    return shadow_status_success;
}


ShadowStatus MixNet::predicModel(const vector<cv::Mat> &imgs, int model_id, InterMethod method)
{
    if (imgs.size() == 0)
    {
        return shadow_status_batchsize_zero_error;
    }
    if (imgs.size() > models[model_id].maxBatchsize)
    {
        cout << "Please input the valid number of images: max = " << models[model_id].maxBatchsize << endl;
        return shadow_status_batchsize_exceed_error;
    }

    processImageGPU(imgs, model_id, method);

    void **buffers = getBuffers(model_id);
    models[model_id].context->enqueue(imgs.size(), buffers, stream, nullptr);
    free(buffers);
    try
    {
        for(size_t i = 0; i < models[model_id].outputInfo.size(); i++)
            cudaMemcpyAsync(models[model_id].outputInfo[i].data, models[model_id].outputInfo[i].data_gpu,imgs.size() * models[model_id].outputInfo[i].dataNum * sizeof(float),cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream);
    }
    catch (...)
    {
        return shadow_status_cuda_memcpy_error;
    }

    return shadow_status_success;
}

void MixNet::saveFeature(vector<FaceBox*> &boxes){
    Info &features = models[2].outputInfo[0];
    for(int i = 0; i < boxes.size(); i++){
        boxes[i]->features = &(features.data[i * features.dataNum]);
    }
}

void MixNet::dealDet3Result(vector<FaceBox*> &boxes, int size){
    Info &points = models[1].outputInfo[0];
    Info &conf = models[1].outputInfo[1];
    for(int i = 0; i < size;i++){
        boxes[i]->conf = conf.data[i * conf.dataNum + 1];
        for(int j = 0; j < 5; j++){
            float x = points.data[i * points.dataNum + j] ;
            float y = points.data[i * points.dataNum + j + 5];
            int weight = boxes[i]->xmax - boxes[i]->xmin;
            int height = boxes[i]->ymax - boxes[i]->ymin;
            cv::Point2d p(x*weight + boxes[i]->xmin, y*height + boxes[i]->ymin);
            boxes[i]->points.push_back(p);
        }
    }
}


void MixNet::dealMixResult(vector<FaceBox> &boxes, vector<vector<int>> &results, int size){    
    Info &detectionOut = models[0].outputInfo[0];
    Info &probBk = models[0].outputInfo[1];
    Info &probPulp = models[0].outputInfo[2];


    for (int k = 0; k < size; k++)
    {
        vector<int> result;
        bool hasFace = false, hasBK = false;
        for (size_t i = k * detectionOut.dataNum; i < (k + 1) * detectionOut.dataNum; i += 7)
        {
            int detectionOutLabel = detectionOut.data[i + 1];
            float conf = detectionOut.data[i + 2];
            if (detectionOutLabel == -1)
            {
                break;
            }
            if (conf > threshold[detectionOutLabel])
            {
                FaceBox box;
                box.xmin = cvRound(originSize[k].first * detectionOut.data[i + 3]);
                box.ymin = cvRound(originSize[k].second * detectionOut.data[i + 4]);
                box.xmax = cvRound(originSize[k].first * detectionOut.data[i + 5]);
                box.ymax = cvRound(originSize[k].second * detectionOut.data[i + 6]);
                box.imageId = k;
                if (detectionOutLabel == 6){
                    boxes.push_back(box);
                    hasFace = true;
                }
                else
                    hasBK = true;
            }
        }
        int bkLabelIndex = GetIndexOfMaxNumber(probBk.data, k * probBk.dataNum, (k + 1) * probBk.dataNum);
        int pulpLabelIndex = GetIndexOfMaxNumber(probPulp.data, k * probPulp.dataNum, (k + 1) * probPulp.dataNum);
        if (hasFace)
            result.push_back(3);
        if (hasBK)
            result.push_back(4);

        if (bkLabel[bkLabelIndex] == "march")
            result.push_back(0);
        else if (bkLabel[bkLabelIndex] == "text")
            result.push_back(2);
        else if (bkLabel[bkLabelIndex] == "terror" && !hasBK)
            result.push_back(4);

        if (pulpLabelIndex == 0 || pulpLabelIndex == 1)
            result.push_back(5);

        if (result.size() == 0)
            result.push_back(1);
        results.push_back(result);
    }
}

string MixNet::getResultJson(vector<int> &result, vector<FaceBox*> &boxes)
{
    Document document;
    auto &alloc = document.GetAllocator();

    Value j_confidences(kObjectType), j_classes(kArrayType);

    int result_array[6] = {0};
    for (int j = 0; j < result.size(); j++)
    {
        result_array[result[j]] = 1;
    }

    Value j_class(kObjectType);
    j_class.AddMember("march", Value(result_array[0]), alloc);
    j_class.AddMember("normal", Value(result_array[1]), alloc);
    j_class.AddMember("text", Value(result_array[2]), alloc);
    j_class.AddMember("face", Value(result_array[3]), alloc);
    j_class.AddMember("bk", Value(result_array[4]), alloc);
    j_class.AddMember("pulp", Value(result_array[5]), alloc);

    Value faces(kArrayType);
    int facenum = 0;
    for(int i = 0; i < boxes.size(); i++){
        if(boxes[i]->conf < 0.7)
            continue;
        facenum++;
        Value face(kObjectType);
        FaceBox rect = *boxes[i];

        Value lt(kArrayType), rt(kArrayType), rb(kArrayType), lb(kArrayType),pts(kArrayType);
        lt.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymin), alloc);
        rt.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymin), alloc);
        rb.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymax), alloc);
        lb.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymax), alloc);
        pts.PushBack(lt,alloc).PushBack(rt,alloc).PushBack(rb,alloc).PushBack(lb,alloc);
        face.AddMember("pts", pts, alloc);

        Value features(kArrayType);
        for(int i = 0;i<128;i++){
            features.PushBack(rect.features[i], alloc);
        }
        face.AddMember("features", features, alloc);
        faces.PushBack(face, alloc);
    }
    j_class.AddMember("facenum", facenum, alloc);
    j_class.AddMember("faces",faces,alloc);

    j_confidences.AddMember("result", j_class, alloc);
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    j_confidences.Accept(writer);
    //j_class.Accept(writer);
    return string(buffer.GetString());
}

ShadowStatus MixNet::destroy()
{
    for(size_t model_id = 0; model_id < models.size(); model_id++){
        Model &model = models[model_id];
        for (size_t i = 0; i < model.inputInfo.size(); i++){
            free(model.inputInfo[i].data);
            cudaFree(model.inputInfo[i].data_gpu);
        }
        for (size_t i = 0; i < model.outputInfo.size(); i++){
            free(model.outputInfo[i].data);
            cudaFree(model.outputInfo[i].data_gpu);
        }
        if (model.context != nullptr)
            model.context->destroy();
        if (model.engine != nullptr)
            model.engine->destroy();
    }
    if (infer != nullptr)
        infer->destroy();
    cudaStreamDestroy(stream);
    pluginFactory.destroyPlugin();
    delete this;
    return shadow_status_success;
}
} // namespace Shadow
