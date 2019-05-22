#include "FdNet.hpp"
#include <time.h>

using namespace cv;

namespace Shadow
{

inline size_t getDimSize(DimsCHW &dim)
{
    return dim.c() * dim.h() * dim.w();
}

inline double clockDiff(double c1, double c2)
{
    return (c1 - c2) / CLOCKS_PER_SEC * 1000;
}

int getIndexOfMax(float *nums, int size){
    float max = 0;
    int index = 0;
    for(int i = 0; i < size; i++){
        if(nums[i] > max){
            max = nums[i];
            index = i;
        }
    }
    return index;
}

FdNet::FdNet(int modelNum_, InterMethod method_) 
{   
    modelNum = modelNum_;
    interMethod = method_;
    engine.resize(modelNum);
    context.resize(modelNum);
    maxBatchSizes.resize(modelNum);
    gProfiler.resize(modelNum);

    fdInputInfo.resize(modelNum);
    fdOutputInfo.resize(modelNum);
    for ( size_t i = 0; i<modelNum; i++)
    {
        engine.at(i) = nullptr;
        context.at(i) = nullptr;
        //buffers.at(i) = nullptr;
        maxBatchSizes.at(i) = 0;
    }
}

//获取输入输出相关信息，并分配显存，
ShadowStatus FdNet::allocateMemory()
{   

    for(int iter = 0; iter < modelNum; iter++)
    {
        int n = engine.at(iter)->getNbBindings();
        //printf("bindings %d\n", n);
        for (int i = 0; i < n; i++)
        {
            Info info;
            info.index = i;
            info.dim = static_cast<DimsCHW &&>(engine.at(iter)->getBindingDimensions(i));
            info.dataNum = getDimSize(info.dim);
            size_t memory = maxBatchSizes.at(0) * info.dataNum * sizeof(float);//using fd index 0 ,malloc bathsize memory for detectionputengin input
            if(!(iter == 1 && engine.at(iter)->bindingIsInput(i))){
                try
                {
                    info.data = (float *)malloc(memory);
                }
                catch (...)
                {
                    return shadow_status_host_malloc_error;
                }
                try
                {
                    cudaMalloc(&info.data_gpu, memory);
                }
                catch (...)
                {
                    return shadow_status_cuda_malloc_error;
                }
            }
            if (engine.at(iter)->bindingIsInput(i))
            {
                //printf("Fd Binding %d (%s): Input. Dims: %d %d %d\n", i, engine.at(iter)->getBindingName(i), info.dim.d[0], info.dim.d[1], info.dim.d[2]);
                fdInputInfo.at(iter).push_back(info);
            }
            else
            {
                //printf("Fd Binding %d (%s): Output. Dims: %d %d %d\n", i, engine.at(iter)->getBindingName(i), info.dim.d[0], info.dim.d[1], info.dim.d[2]);
                fdOutputInfo.at(iter).push_back(info);
            }
        }
    }
    return shadow_status_success;
}

//初始化网络
ShadowStatus FdNet::init(const int gpuID, const std::vector<std::vector<char>> data,const std::vector<int> size)
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
        for(int i = 0; i < modelNum; i++)
        {
            engine.at(i) = infer->deserializeCudaEngine((void*)(data.at(i).data()), size.at(i), &pluginFactory);
            context.at(i) = engine.at(i)->createExecutionContext();
            context.at(i)->setProfiler(&(gProfiler.at(i)));
            maxBatchSizes.at(i) = engine.at(i)->getMaxBatchSize();
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
ShadowStatus FdNet::init(const int gpuID, void *data, int size)
{
    return shadow_status_success;
}

void FdNet::processImageGPU(const vector<cv::Mat> &imgs, int model_id, float *preParams)
{
    DimsCHW dim = fdInputInfo.at(model_id)[0].dim;
    int dataNum = fdInputInfo.at(model_id)[0].dataNum;
    for (size_t b = 0; b < imgs.size(); b++)
    {
        cv::Mat img = imgs[b];
        inputGpuMat.upload(img);
        process(inputGpuMat, (float *)(fdInputInfo.at(model_id)[0].data_gpu) + b * dataNum,
                dim.w(), dim.h(), preParams, interMethod);
    }
}

ShadowStatus FdNet::predict(const vector<cv::Mat> &imgs, const std::vector<string> &attributes, std::vector<string> &results)
{
    originSize.clear();
    for(int i = 0; i < imgs.size(); i++){
        originSize.push_back(make_pair(imgs[i].cols, imgs[i].rows));
    }
    //Face detection
    if (imgs.size() == 0)
    {
        return shadow_status_batchsize_zero_error;
    }
    if (imgs.size() > maxBatchSizes.at(0))
    {
        cout << "Please input the valid number of images: max = " << maxBatchSizes.at(0) << endl;
        return shadow_status_batchsize_exceed_error;
    }
    //double pre_time_start = clock();  
    processImageGPU(imgs, 0, fdPreParams);
    //pre_time += clockDiff(clock(), pre_time_start);
    
    //infer processing
    //double infer_time_start = clock();
    void **buffers = getBuffers(0);
    context.at(0)->enqueue(imgs.size(), buffers, stream, nullptr);
    for(size_t i = 0; i < imgs.size(); i++){
        buffers[fdInputInfo.at(1)[0].index]  = fdOutputInfo.at(0)[0].data_gpu + i * fdOutputInfo.at(0)[0].dataNum * sizeof(float);
        buffers[fdInputInfo.at(1)[1].index]  = fdOutputInfo.at(0)[1].data_gpu + i * fdOutputInfo.at(0)[1].dataNum * sizeof(float);
        buffers[fdInputInfo.at(1)[2].index]  = fdOutputInfo.at(0)[2].data_gpu + i * fdOutputInfo.at(0)[2].dataNum * sizeof(float);
        buffers[fdOutputInfo.at(1)[0].index] = fdOutputInfo.at(1)[0].data_gpu + i * fdOutputInfo.at(1)[0].dataNum * sizeof(float);
        context.at(1)->enqueue(1, buffers, stream, nullptr);
    }
    try
    {
        cudaMemcpyAsync(fdOutputInfo.at(1)[0].data, fdOutputInfo.at(1)[0].data_gpu,
                        imgs.size() * fdOutputInfo.at(1)[0].dataNum * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    catch (...)
    {
        return shadow_status_cuda_memcpy_error;
    }
    cudaStreamSynchronize(stream);
    free(buffers);
    //infer_time += clockDiff(clock(), infer_time_start);

    vector<vector<FdResult>> faceBoxes;
    vector<cv::Mat> faces;
    getFaceBox(imgs.size(), faceBoxes);
    cropFace(imgs, faceBoxes, faces);
    
    int imgId = 0, faceId = 0;
    int round = faces.size() / maxBatchSizes[2];
    for(int i = 0; i < round; i++){
        vector<cv::Mat> faceBatch(faces.begin() + i * maxBatchSizes[2], faces.begin() + (i + 1) * maxBatchSizes[2]);
        predictOnet(faceBatch);
        dealResultOnet(faceBatch.size(), faceBoxes, imgId, faceId);

    }
    if(faces.size() % maxBatchSizes[2] != 0){
        vector<cv::Mat> faceBatch(faces.begin() + round * maxBatchSizes[2], faces.end());
        predictOnet(faceBatch);
        dealResultOnet(faceBatch.size(), faceBoxes, imgId, faceId);
    }

    //draw results
    //drawResult(imgs, attributes, faceBoxes);
    for(int i = 0; i < faceBoxes.size(); i++)
        results.push_back(getDetectJson(true,true,faceBoxes[i]));

    return shadow_status_success;
}

void FdNet::cropFace(vector<cv::Mat> imgs, vector<vector<FdResult>> &faceBoxes, std::vector<cv::Mat> &faces){
    float scale = 0.0143;
    for(int i = 0; i < imgs.size(); i++){
        for(int j = 0; j < faceBoxes[i].size(); j++){
            FdResult &faceBox = faceBoxes[i][j];
            int width = faceBox.xmax - faceBox.xmin + 1;
            int height = faceBox.ymax - faceBox.ymin + 1;
            faceBox.xmin = std::max(0, cvRound(faceBox.xmin - scale * width));
            faceBox.ymin = std::max(0, cvRound(faceBox.ymin - scale * height));
            faceBox.xmax = std::min(originSize[i].first  - 1, cvRound(faceBox.xmax + scale * width));
            faceBox.ymax = std::min(originSize[i].second - 1, cvRound(faceBox.ymax + scale * height));
            faceBoxes[i][j] = faceBox;
            width = faceBox.xmax - faceBox.xmin + 1;
            height = faceBox.ymax - faceBox.ymin + 1;
            if(width < 50 || height < 50){
                faceBox.quality_category = 5;//small
            }
            cv::Rect rect(faceBox.xmin, faceBox.ymin, width, height);
            cv::Mat face(imgs[i], rect);
            faces.push_back(face);
        }
    }
}

void **FdNet::getBuffers(int model_id){
    int n = engine.at(model_id)->getNbBindings();
    void **buffers = (void**)malloc(sizeof(void*) * n);
    for(int i = 0; i < fdInputInfo.at(model_id).size(); i++)
        buffers[fdInputInfo.at(model_id)[i].index] = fdInputInfo.at(model_id)[i].data_gpu;
    for(int i = 0; i < fdOutputInfo.at(model_id).size(); i++)
        buffers[fdOutputInfo.at(model_id)[i].index] = fdOutputInfo.at(model_id)[i].data_gpu;
    return buffers;
}

void FdNet::predictOnet(const std::vector<cv::Mat> faces){
    processImageGPU(faces, 2, onetPreParams);
    void **buffers = getBuffers(2);
    context.at(2)->enqueue(faces.size(), buffers, stream, nullptr);
    for (size_t i = 0; i < fdOutputInfo.at(2).size(); i++)
    {
        cudaMemcpyAsync(fdOutputInfo.at(2)[i].data, fdOutputInfo.at(2)[i].data_gpu,
                        faces.size() * fdOutputInfo.at(2)[i].dataNum * sizeof(float), cudaMemcpyDeviceToHost, stream);

    }
    cudaStreamSynchronize(stream);
    free(buffers);
}

string FdNet::getDetectJson(bool output_quality,bool output_quality_score, const vector<FdResult> detection_output){
    using namespace rapidjson;
    Document document;
    auto &alloc = document.GetAllocator();
    Value j_detections(kObjectType), j_rects(kArrayType);
    for (const auto &rect : detection_output) {
        if(rect.quality_category != 1) {
            Value j_rect(kObjectType);
            j_rect.AddMember("index", Value(1), alloc);

            j_rect.AddMember("score", Value(rect.conf), alloc);

            j_rect.AddMember("class", Value("face"), alloc);

            Value lt(kArrayType), rt(kArrayType), rb(kArrayType), lb(kArrayType),pts(kArrayType);
            lt.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymin), alloc);
            rt.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymin), alloc);
            rb.PushBack(Value(rect.xmax), alloc).PushBack(Value(rect.ymax), alloc);
            lb.PushBack(Value(rect.xmin), alloc).PushBack(Value(rect.ymax), alloc);
            pts.PushBack(lt,alloc).PushBack(rt,alloc).PushBack(rb,alloc).PushBack(lb,alloc);
            j_rect.AddMember("pts", pts, alloc);

            if(output_quality){
                if(rect.quality_category==0)
                    j_rect.AddMember("quality", Value("clear"), alloc);
                if(rect.quality_category==2)
                    j_rect.AddMember("quality", Value("blur"), alloc);
                if(rect.quality_category==3)
                    j_rect.AddMember("quality", Value("pose"), alloc);
                if(rect.quality_category==4)
                    j_rect.AddMember("quality", Value("cover"), alloc);
                if(rect.quality_category==5)
                    j_rect.AddMember("quality", Value("small"), alloc);
                
                switch(rect.orient_category) {
                case 0:
                    j_rect.AddMember("orientation", Value("up"), alloc);
                    break;
                case 1:
                    j_rect.AddMember("orientation", Value("up_left"), alloc);
                    break;
                case 2:
                    j_rect.AddMember("orientation", Value("left"), alloc);
                    break;
                case 3:
                    j_rect.AddMember("orientation", Value("down_left"), alloc);
                    break;
                case 4:
                    j_rect.AddMember("orientation", Value("down"), alloc);
                    break;
                case 5:
                    j_rect.AddMember("orientation", Value("down_right"), alloc);
                    break;
                case 6:
                    j_rect.AddMember("orientation", Value("right"), alloc);
                    break;
                case 7:
                    j_rect.AddMember("orientation", Value("up_right"), alloc);
                    break;
                }
            }
            if(output_quality_score){
                Value q_score(kObjectType);
                q_score.AddMember("clear", Value(rect.quality_cls[0]), alloc);
                q_score.AddMember("blur", Value(rect.quality_cls[2]), alloc);
                q_score.AddMember("neg", Value(rect.quality_cls[1]), alloc);
                q_score.AddMember("cover", Value(rect.quality_cls[4]), alloc);
                q_score.AddMember("pose", Value(rect.quality_cls[3]), alloc);
                j_rect.AddMember("q_score", q_score, alloc);
            }
            j_rects.PushBack(j_rect, alloc);
        }
    }
    j_detections.AddMember("detections", j_rects, alloc);
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    j_detections.Accept(writer);
    return std::string(buffer.GetString());
}


void FdNet::dealResultOnet(int size, vector<vector<FdResult>> &faceBoxes, int &imgId, int &faceId){
    int quality_num = fdOutputInfo.at(2)[0].dataNum;
    int orient_num = fdOutputInfo.at(2)[1].dataNum;
    float *quality_scores = fdOutputInfo.at(2)[0].data;
    float *orient_scores  = fdOutputInfo.at(2)[1].data;
    for(int i = 0; i < size; i++){
        while(faceId == faceBoxes[imgId].size()){
            imgId++;
            faceId = 0;
        }
        FdResult &faceBox = faceBoxes[imgId][faceId];
        memcpy(faceBox.quality_cls, quality_scores + i * quality_num, quality_num * sizeof(float));
        int quality_category = getIndexOfMax(faceBox.quality_cls , quality_num);
        if(faceBox.quality_category != 5){
            faceBox.quality_category = quality_category;
            if(quality_category == 0 && faceBox.quality_cls[0] < 0.6){//processing clear
                float clear_conf = faceBox.quality_cls[0];
                faceBox.quality_cls[0] = 0;
                int second_category = getIndexOfMax(faceBox.quality_cls , quality_num);
                faceBox.quality_cls[0] = clear_conf;
                faceBox.quality_category = second_category;
            }
        }
        faceBox.orient_category = getIndexOfMax(orient_scores + i * orient_num, orient_num);
        faceId++;
    }
}

void FdNet::getFaceBox(int size, vector<vector<FdResult> > &batchResults){   
    batchResults.clear();
    size_t outputSize = fdOutputInfo.at(1)[0].dataNum;
    float *detectionOut = fdOutputInfo.at(1)[0].data;
    for (size_t k = 0; k < size; k++)
    {
        vector<FdResult> results;
        for (size_t i = 0; i < outputSize; i += 7)
        {
            int offset = k * outputSize + i;
            if (detectionOut[offset + 1] == -1)
                break;
            FdResult result;
            result.conf = detectionOut[offset + 2];
            result.xmin = cvRound(originSize[k].first * detectionOut[offset + 3]);
            result.ymin = cvRound(originSize[k].second * detectionOut[offset + 4]);
            result.xmax = cvRound(originSize[k].first * detectionOut[offset + 5]);
            result.ymax = cvRound(originSize[k].second * detectionOut[offset + 6]);
            results.push_back(result);
        }
        batchResults.push_back(results);
    }
}

void FdNet::drawResult(const vector<cv::Mat> &imgs, const vector<string> &attributes, vector<vector<FdResult> > &batchResults)
{
    int face_id = 0;
    const char *qualityStr[6] = {"clear", "neg", "blur", "pose", "cover", "small"};
    const char *orientationStr[8] = {"up", "up_left", "left", "down_left", "down", "down_right", "right", "up_right"};
    for (int i = 0; i < imgs.size(); i++)
    {
        Mat cpImage = Mat(imgs[i]);
        std::cout<<"Detecte face: "<<batchResults[i].size()<<std::endl;
        for (int j = 0; j < batchResults[i].size(); j++)
        {
            FdResult &result = batchResults[i][j];
            cv::putText(cpImage, string(qualityStr[result.quality_category]), Point2f(result.xmin, result.ymin + 15), FONT_HERSHEY_DUPLEX, 1, (0, 253, 255), 2, 1);
            cv::putText(cpImage, string(orientationStr[result.orient_category]), Point2f(result.xmin, result.ymin + 55), FONT_HERSHEY_DUPLEX, 1, (0, 253, 255), 2, 1);
            cv::rectangle(cpImage, Point(result.xmin, result.ymin), Point(result.xmax, result.ymax), Scalar(0, 0, 255));
        }
        cv::imwrite(attributes[i], cpImage);
    }
}


ShadowStatus FdNet::destroy()
{
    // relese the input and output data
    for(int model_id = 0; model_id < modelNum; model_id++){
        for (size_t i = 0; i < fdInputInfo.at(model_id).size(); i++){
            if(model_id != 1){
                free(fdInputInfo.at(model_id)[i].data);
                cudaFree(fdInputInfo.at(model_id)[i].data_gpu);
            }
        }
        for (size_t i = 0; i < fdOutputInfo.at(model_id).size(); i++){
            free(fdOutputInfo.at(model_id)[i].data);
            cudaFree(fdOutputInfo.at(model_id)[i].data_gpu);
        }
        if (context.at(model_id) != nullptr)
            context.at(model_id)->destroy();
        if (engine.at(model_id) != nullptr)
            engine.at(model_id)->destroy();

    }

    if (infer != nullptr)
        infer->destroy();
    cudaStreamDestroy(stream);
    pluginFactory.destroyPlugin();
    return shadow_status_success;
}

} // namespace Shadow
