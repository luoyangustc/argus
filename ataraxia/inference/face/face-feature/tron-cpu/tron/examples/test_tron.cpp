#include "serving/infer_algorithm.hpp"
#include "proto/inference.pb.h"
#include "common/util.hpp"
//#include"opencv2/opencv.hpp"

using namespace std;

int main(int argc, char const *argv[]) {

  string model_mtcnn("models/mtcnn_merged.tronmodel");
  string model_feature("models/resnet34_v1.1_merged.tronmodel");

  cout << "-----------------------------------------------------" << endl;
  cout << "USAGE: " << endl;
  cout << argv[0] << " <image-path> <face-pts-string>" 
       << " --mtcnn-model <mtcnn-tronmodel-path> --feature-model <feature-model-path>" << endl;
  cout << "<image-path>: path to image file" << endl;
  cout << "<face-pts-string>: a string in the format as follows:" << endl;
  cout << "    '[[20,30],[100,30],[100,110],[20,110]]' " << endl;
  cout << "<mtcnn-tronmodel-path>: [optional] path to mtcnn tronmodel" << endl;
  cout << "<feature-tronmodel-path>: [optional] path to feature tronmodel" << endl;
  cout << "-----------------------------------------------------" << endl << endl;

  if (argc!=3 && argc!=5 && argc!=7)
  {
    cout << "Wrong number of argv!!! exit..." << endl;
    return -1;
  }

  if (argc>3)
  {
    for (int i=3;  i<argc; i+=2)
    {
      cout << "argv[" << i << "]: " << argv[i]  << endl;
      string t_str(argv[i]);
      if (t_str=="--mtcnn-model")
      {
        model_mtcnn = argv[i+1];
      }
      else if (t_str=="--feature-model")
      {
        model_feature = argv[i+1];
      }
    }
  }

  cout << "-----------------------------------------------------" << endl;
  cout << "mtcnn-model: " << model_mtcnn << endl;
  cout << "feature-model: " << model_feature << endl;
  cout << "input image: " << argv[1] << endl;
  cout << "face pts: " << argv[2] << endl;
  cout << "-----------------------------------------------------" << endl << endl;

  //string im_file("data/static/001.png");
  string im_file(argv[1]);

 // cv::Mat src=cv::imread("data/static/001.jpg");
 // imwrite(im_file,src);
  Tron::Timer timer;
  double time_cost = 0;
  int loop_count = 1;
 // cout<<"before model_mtcnn_fp\n";
  // Read model from file to bytes
  auto *model_mtcnn_fp = fopen(model_mtcnn.c_str(), "rb");
  fseek(model_mtcnn_fp, 0, SEEK_END);
  auto model_mtcnn_size = ftell(model_mtcnn_fp);
//  cout<<"before rewind\n";
  rewind(model_mtcnn_fp);
//  cout<<"after rewind\n";
  vector<char> model_mtcnn_data(model_mtcnn_size, 0);
  fread(model_mtcnn_data.data(), 1, model_mtcnn_size, model_mtcnn_fp);
  fclose(model_mtcnn_fp);

// cout<<"before model_feature_fp\n";
  // Read model from file to bytes
  auto *model_feature_fp = fopen(model_feature.c_str(), "rb");
  fseek(model_feature_fp, 0, SEEK_END);
  auto model_feature_size = ftell(model_feature_fp);
  rewind(model_feature_fp);
  vector<char> model_feature_data(model_feature_size, 0);
  fread(model_feature_data.data(), 1, model_feature_size, model_feature_fp);
  fclose(model_feature_fp);

// cout<<"before create_params\n";
  // Prepare protobuf's CreateParams
  inference::CreateParams create_params;
  auto *model_mtcnn_file = create_params.add_model_files();
  model_mtcnn_file->set_name(model_mtcnn.c_str());
  model_mtcnn_file->set_body(model_mtcnn_data.data(), model_mtcnn_size);
  auto *model_feature_file = create_params.add_model_files();
  model_feature_file->set_name(model_feature.c_str());
  model_feature_file->set_body(model_feature_data.data(), model_feature_size);

  create_params.set_custom_params(R"({"gpu_id": 0})");

  // Serialize protobuf's CreateParams to bytes
  auto create_params_size = create_params.ByteSize();
  vector<char> create_params_data(create_params_size, 0);
  create_params.SerializeToArray(create_params_data.data(), create_params_size);

  int code = -1;
  char *err = nullptr;
  
  cout << "\n---> createNet() starts" << endl; 
  // Do createNet
  auto *handle =createNet(create_params_data.data(), create_params_size, &code, &err);
//  cout<<"after createNet.\n";
  cout << "<--- createNet() finished" << endl;

  if (code == 200) {
    for (int i = 0; i < loop_count; ++i) {
      // Read image from file to bytes
      auto *im_fp = fopen(im_file.c_str(), "rb");
      fseek(im_fp, 0, SEEK_END);
      auto im_size = ftell(im_fp);
      rewind(im_fp);
      vector<char> image_data(im_size, 0);
      fread(image_data.data(), 1, im_size, im_fp);
      fclose(im_fp);

      // Prepare protobuf's InferenceRequests
      inference::InferenceRequests inference_requests;
      auto *request = inference_requests.add_requests();
      
      request->mutable_data()->set_uri(im_file);
      request->mutable_data()->set_body(image_data.data(), im_size);

      string pts=R"({"pts":)";
      pts += argv[2];
      pts += "}";

      request->mutable_data()->set_attribute(pts);

      // Serialize protobuf's InferenceRequests to bytes
      auto inference_requests_size = inference_requests.ByteSize();
      vector<char> inference_requests_data(inference_requests_size, 0);
      inference_requests.SerializeToArray(inference_requests_data.data(),
                                          inference_requests_size);

      // Do netInference
      vector<char> inference_responses_data(4 * 1024 * 1024, 0);
      int inference_responses_size;
    
      timer.start();
      cout<<"\n---> netInference() starts" << endl;
      netInference(handle, inference_requests_data.data(),
                   inference_requests_size, &code, &err,
                   inference_responses_data.data(), &inference_responses_size);
      time_cost += timer.get_millisecond();
      cout<<"<--- netInference() finished" << endl;
    
      cout << "response code: " << code << endl;
      cout << "response err info: " << err << endl;

      if (code == 200) {
        // Parse protobuf's InferenceResponses from bytes
        inference::InferenceResponses inference_responses;
        inference_responses.ParseFromArray(inference_responses_data.data(),
                                           inference_responses_size);
   
        float features[512];
        memcpy(features, inference_responses.responses(0).body().c_str(),512*sizeof(float) );

        for(int i=0;i<512;++i){
          cout<<features[i]<<" ";
        }
        cout << endl;

      } else {
        cout << err << endl;
      }
    }

    cout << "time cost: " << time_cost / loop_count<<" ms..." << endl;
  } else {
    cout << err << endl;
  }

  return 0;
}
