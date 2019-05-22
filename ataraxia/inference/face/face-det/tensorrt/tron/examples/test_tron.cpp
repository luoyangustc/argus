#include "serving/infer_algorithm.hpp"

#include "proto/inference.pb.h"

#include "common/util.hpp"
#include <string>
#include <vector>
using namespace std;

//获取文件里面所有图片的名字
void GetImageName(const char *fileName, vector<string> &imageName){
    FILE* f = fopen(fileName, "r");
	if(f == NULL){
		cerr << "can not open image_list file" << endl;
		exit(-1);
	}
    char buffer[300];
    while(fgets(buffer, 300, f)){
        //去掉换行符
        buffer[strlen(buffer)-1] = '\0';
        imageName.push_back(string(buffer));
    }

}

void getModel(vector<vector<char>> &models, vector<int> &modelSize, int model_id, const char *engineFilename){
    FILE *file = fopen(engineFilename, "rb");
    if (!file)
    {
        cerr << "can not open engine file" << endl;
        exit(-1);
    }
    fseek(file, 0, SEEK_END);
    int enginSize = ftell(file);
    rewind(file);
    models.at(model_id).resize(enginSize);
    fread((void*)models.at(model_id).data(), 1, enginSize, file);
    modelSize.at(model_id) = enginSize;
    fclose(file);
}

int main(int argc, char const *argv[]) {
  if(argc != 6){
    std::cout<<"Usage:"<<std::endl;
    std::cout<<"./build/tron/test_tron models/fdnet_engin.bin models/dptnet_engin.bin models/onet_engin.bin images/image_list.list 8"<<std::endl;
    return -1;
  }

  std::cout<<"start testing..."<<std::endl;
  std::string fdnet_model_file(argv[1]);
  std::string dptnet_model_file(argv[2]);
  std::string onet_model_file(argv[3]);
  
  std::vector<std::string> im_list;
  std::cout<<"start load data..."<<std::endl;
  GetImageName(argv[4], im_list);
  
  int batchSize = atoi(argv[5]);
  Tron::Timer timer;
  double time_cost = 0;
  
  int modelNum = 3; 
  vector<vector<char>> models;
  models.resize(modelNum);
  vector<int> modelSize;
  modelSize.resize(modelNum);

  getModel(models, modelSize, 0, fdnet_model_file.c_str());
  getModel(models, modelSize, 1, dptnet_model_file.c_str());
  getModel(models, modelSize, 2, onet_model_file.c_str());

  inference::CreateParams create_params;
  auto* model_files = create_params.add_model_files();
  model_files->set_name(fdnet_model_file.c_str());
  model_files->set_body(models[0].data(), modelSize[0]);
  
  model_files = create_params.add_model_files();
  model_files->set_name(dptnet_model_file.c_str());
  model_files->set_body(models[1].data(), modelSize[1]);

  model_files = create_params.add_model_files();
  model_files->set_name(onet_model_file.c_str());
  model_files->set_body(models[2].data(), modelSize[2]);

  //custom very important
  const std::string custom_params=R"({"gpu_id": 0,"model_num": 3})";
  create_params.set_custom_params(custom_params);

  auto create_params_size = create_params.ByteSize();
  std::vector<char> create_params_data(create_params_size, 0);
  create_params.SerializeToArray(create_params_data.data(), create_params_size);

  int code = -1;
  char *err = nullptr;
  auto *handle = createNet(create_params_data.data(), create_params_size, &code, &err);
  std::cout<<"Total images: "<<im_list.size()<<std::endl;
  int currCount = 0;
  if (code == 200) {
    int rounds = im_list.size()/batchSize;
    int rest = im_list.size()%batchSize;
    int count=0;
    for (int i = 0; i < rounds; i++) {
      inference::InferenceRequests inference_requests;
      for(int j= 0; j < batchSize; j++) {
        std::string im_file(im_list[count++]);
        auto *im_fp = fopen(im_file.c_str(), "rb");
        fseek(im_fp, 0, SEEK_END);
        auto im_size = ftell(im_fp);
        rewind(im_fp);
        std::vector<char> image_data(im_size, 0);
        fread(image_data.data(), 1, im_size, im_fp);
        fclose(im_fp);
        auto *request = inference_requests.add_requests();
        request->mutable_data()->set_body(image_data.data(), im_size);
        currCount++;
      }

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
      std::cout <<"\n\n Iter "<<currCount<< " time cost per: " << timer.get_millisecond()/ batchSize <<" ms\n";
      if (code == 200) {
        inference::InferenceResponses inference_responses;
        inference_responses.ParseFromArray(inference_responses_data.data(),
                                           inference_responses_size);
        for(int i = 0; i < inference_responses.responses_size();i++)
        {
            std::cout <<inference_responses.responses(i).code()<<" "<<inference_responses.responses(i).message()<<" "<< inference_responses.responses(i).result() << std::endl;
        }
      } else {
        std::cout << err << std::endl;
      }
    }
    //rest
    inference::InferenceRequests inference_requests;
    for(int j= 0; j < rest; j++) {
        std::string im_file(im_list[count++]);
        auto *im_fp = fopen(im_file.c_str(), "rb");
        fseek(im_fp, 0, SEEK_END);
        auto im_size = ftell(im_fp);
        rewind(im_fp);
        std::vector<char> image_data(im_size, 0);
        fread(image_data.data(), 1, im_size, im_fp);
        fclose(im_fp);
        auto *request = inference_requests.add_requests();
        request->mutable_data()->set_body(image_data.data(), im_size);
        currCount++;
      }

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
      std::cout <<"\n\nIter "<<currCount<< "time cost per: " << timer.get_millisecond()/ batchSize <<" ms\n";
      if (code == 200) {
        inference::InferenceResponses inference_responses;
        inference_responses.ParseFromArray(inference_responses_data.data(),
                                           inference_responses_size);
        for(int i = 0; i < inference_responses.responses_size();i++)
        {
            std::cout <<inference_responses.responses(i).code()<<" "<<inference_responses.responses(i).message()<<" "<< inference_responses.responses(i).result() << std::endl;
        }
      } else {
        std::cout << err << std::endl;
      }

  }
  return 0;
}
