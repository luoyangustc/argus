#include "serving/infer_algorithm.hpp"
#include "proto/inference.pb.h"
#include "common/util.hpp"

int main(int argc, char const *argv[]) {
    std::cout<<"start testing..."<<std::endl;
    std::string model_fd_file("data/models/refinedet_v0.0.2.tronmodel");
    std::string model_quality_file("data/models/quality_v0.0.2.tronmodel");
    std::vector<std::string> im_list;
    //std::string im_file("../data/001.jpg");
    std::cout<<"start load data..."<<std::endl;
    std::string img_path;
    int loop_count = 0;
    for (int i = 1; i < 27; i++) {
        img_path = "data/";
        char num[10];
        sprintf(num, "%03d", i);
        std::string temp(num);
        img_path += temp;
        img_path += ".jpg";
        im_list.push_back(img_path);
        loop_count++;
        //std::cout<<img_path<<std::endl;
    }
    
    Tron::Timer timer;
    double time_cost = 0;
    
    
    auto *model_fd_fp = fopen(model_fd_file.c_str(), "rb");
    fseek(model_fd_fp, 0, SEEK_END);
    auto model_fd_size = ftell(model_fd_fp);
    rewind(model_fd_fp);
    std::vector<char> model_fd_data(model_fd_size, 0);
    fread(model_fd_data.data(), 1, model_fd_size, model_fd_fp);
    fclose(model_fd_fp);
    
    // read quality model
    auto *model_quality_fp = fopen(model_quality_file.c_str(), "rb");
    fseek(model_quality_fp, 0, SEEK_END);
    auto model_quality_size = ftell(model_quality_fp);
    rewind(model_quality_fp);
    std::vector<char> model_quality_data(model_quality_size, 0);
    fread(model_quality_data.data(), 1, model_quality_size, model_quality_fp);
    fclose(model_quality_fp);
    
    inference::CreateParams create_params;
    auto* model_files = create_params.add_model_files();
    model_files->set_name(model_fd_file.c_str());
    model_files->set_body(model_fd_data.data(), model_fd_size);
    
    // quality
    model_files = create_params.add_model_files();
    model_files->set_name(model_quality_file.c_str());
    model_files->set_body(model_quality_data.data(), model_quality_size);
    
    const std::string custom_params=R"({"gpu_id": 0,"const_use_quality": 1,"output_quality_score": 1,"blur_threshold": 0.98, "min_face": 50})";
    create_params.set_custom_params(custom_params);
    auto create_params_size = create_params.ByteSize();
    std::vector<char> create_params_data(create_params_size, 0);
    create_params.SerializeToArray(create_params_data.data(), create_params_size);
    
    int code = -1;
    char *err = nullptr;
    auto *handle = createNet(create_params_data.data(), create_params_size, &code, &err);
    std::cout<<code<<" "<<err<<std::endl;
    std::string request_params = R"({"use_quality":0})";
    if (code == 200) {
            for (int i = 0; i < loop_count; i++) {
            std::string im_file(im_list[i]);
            std::cout<<im_file<<std::endl;
            auto *im_fp = fopen(im_file.c_str(), "rb");
            fseek(im_fp, 0, SEEK_END);
            auto im_size = ftell(im_fp);
            rewind(im_fp);
            std::vector<char> image_data(im_size, 0);
            fread(image_data.data(), 1, im_size, im_fp);
            fclose(im_fp);
            
            inference::InferenceRequests inference_requests;
            auto *request = inference_requests.add_requests();
            request->mutable_data()->set_body(image_data.data(), im_size);
            request->set_params(request_params);

            auto inference_requests_size = inference_requests.ByteSize();
            std::vector<char> inference_requests_data(inference_requests_size, 0);
            inference_requests.SerializeToArray(inference_requests_data.data(),
                                                inference_requests_size);
            
            std::vector<char> inference_responses_data(4 * 1024 * 1024, 0);
            
            timer.start();
            int inference_responses_size; //
            netInference(handle, inference_requests_data.data(),
                         inference_requests_size, &code, &err,
                         inference_responses_data.data(), &inference_responses_size);
            time_cost += timer.get_millisecond();
            if (code == 200) {
                inference::InferenceResponses inference_responses;
                inference_responses.ParseFromArray(inference_responses_data.data(),
                                                   inference_responses_size);
                
                std::cout << inference_responses.responses(0).result() << std::endl;
            } else {
                std::cout <<code<<" "<< err << std::endl;
            }
        }
        std::cout << "time cost: " << time_cost / loop_count <<" ms\n";
    } else {
        std::cout << err << std::endl;
    }
    
    return 0;
}
