#include "serving/infer_algorithm.hpp"

#include "proto/inference.pb.h"

#include "common/util.hpp"

int main(int argc, char const *argv[]) {
  std::cout<<"start testing..."<<std::endl;
  std::string model_file("models/refinedet/refinedet_v1.1.tronmodel");
  std::vector<std::string> im_list;
  //std::string im_file("../data/001.jpg");
  std::cout<<"start load data..."<<std::endl;
  std::string img_path;
  for (int i = 1; i < 31; i++) {
    img_path = "data/images/";
    char num[10];
    sprintf(num, "%03d", i);
    std::string temp(num);
    img_path += temp;
    img_path += ".jpg";
    im_list.push_back(img_path);
    //std::cout<<img_path<<std::endl;
  }

  Tron::Timer timer;
  double time_cost = 0;
  int loop_count = 30;

  auto *model_fp = fopen(model_file.c_str(), "rb");
  fseek(model_fp, 0, SEEK_END);
  auto model_size = ftell(model_fp);
  rewind(model_fp);
  std::vector<char> model_data(model_size, 0);
  fread(model_data.data(), 1, model_size, model_fp);
  fclose(model_fp);

  inference::CreateParams create_params;
  auto *model_files = create_params.add_model_files();
  model_files->set_name("dcn_rfcn_merged.shadowmodel");
  model_files->set_body(model_data.data(), model_size);
  create_params.set_custom_params("{\"gpu_id\": 0}");

  auto create_params_size = create_params.ByteSize();
  std::vector<char> create_params_data(create_params_size, 0);
  create_params.SerializeToArray(create_params_data.data(), create_params_size);

  int code = -1;
  char *err = nullptr;

  auto *handle =
      createNet(create_params_data.data(), create_params_size, &code, &err);

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

      auto inference_requests_size = inference_requests.ByteSize();
      std::vector<char> inference_requests_data(inference_requests_size, 0);
      inference_requests.SerializeToArray(inference_requests_data.data(),
                                          inference_requests_size);

      std::vector<char> inference_responses_data(4 * 1024 * 1024, 0);

      timer.start();
      int inference_responses_size; //
      /*auto *inference_responses_data = netInference(
          handle, inference_requests_data.data(), inference_requests_size,
          &code, &err, &inference_responses_size);//*/
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
        std::cout << err << std::endl;
      }
    }
    std::cout << "time cost: " << time_cost / loop_count << std::endl;
  } else {
    std::cout << err << std::endl;
  }

  return 0;
}
