#include "serving/infer_algorithm.hpp"

#include "proto/inference.pb.h"

#include "common/util.hpp"
#include<iostream>
#include<fstream>
#include<opencv2/opencv.hpp>
using namespace std;

inline char *strsep(char **stringp, const char *delim) {
    char *s;
    const char *spanp;
    int c, sc;
    char *tok;
    if ((s = *stringp)== NULL)
        return (NULL);
    for (tok = s;;) {
        c = *s++;
        spanp = delim;
        do {
            if ((sc =*spanp++) == c) {
                if (c == 0)
                    s = NULL;
                else
                    s[-1] = 0;
                *stringp = s;
                return (tok);
            }
        } while (sc != 0);
    }
}

void pushToVec(std::vector<std::string>&obj,const char *param,std::string token){
    char *p = (char*)param;
    char *key_point;
    while(p)
    {
        while ( key_point = strsep(&p,token.c_str()))//关键字为c或d，它们连续出现了
        {
            if (*key_point == 0)
                continue;
            else
                break;
        }
        if (key_point)
            obj.push_back(key_point);
    }
}

template <class Type>
inline Type stringToNum(const string& str){
    istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

int main(int argc, char const *argv[]) {
  std::cout<<"start testing..."<<std::endl;
  std::string model_file("model_tron/mobilenetv2_merged.tronmodel");
  cout << "-----------------------------------------------------" << endl;
  auto *model_fp = fopen(model_file.c_str(), "rb");
  fseek(model_fp, 0, SEEK_END);
  auto model_size = ftell(model_fp);
  rewind(model_fp);
  std::vector<char> model_data(model_size, 0);
  fread(model_data.data(), 1, model_size, model_fp);
  fclose(model_fp);

  inference::CreateParams create_params;
  auto *model_files = create_params.add_model_files();
  model_files->set_name("mobilenetv2_merged.shadowmodel");
  model_files->set_body(model_data.data(), model_size);
  create_params.set_custom_params("{\"gpu_id\": 0}");

  auto create_params_size = create_params.ByteSize();
  std::vector<char> create_params_data(create_params_size, 0);
  create_params.SerializeToArray(create_params_data.data(), create_params_size);

  int code = -1;
  char *err = nullptr;

  auto *handle = createNet(create_params_data.data(),create_params_size,&code,&err);


  if (code == 200) {
  std::cout<<"start load data..."<<std::endl;
  std::string img_path("");

  const string str_img_list="images/filelist.txt";
  ifstream fin(str_img_list);
  string str_name;
  int loop_count = 0;
  while(getline(fin,str_name)){
        cout << "Read from file: " << str_name << endl;
        string str_img="images/"+str_name+".jpg";
        string str_img_dep="images/"+str_name+"_deploy.jpg";
        ++loop_count;
        string str_rect_txt="images/"+str_name+".txt";
        std::vector<std::string> vec_pts;
        fstream file;
        int ch;
        file.open(str_rect_txt.c_str(),ios::in);
        ch=file.get();
        if(file.eof())
        {
           cout<<"文件为空"<<endl;
           vec_pts.push_back("");
        }else
        {
          ifstream fin_rect(str_rect_txt);
          const string token(",");
          string str_line(""),str_pts("");
          int line=0;
          while(getline(fin_rect,str_line)){
             if(line==0)
               str_pts += R"({"pts":[)";
             else
               str_pts += R"(,{"pts":[)";
             line++;
             vector<string> obj;
             pushToVec(obj,str_line.c_str(),token);
             stringstream ss;
             float temp[4];float pts[4][2];
             temp[0]=stringToNum<float>(obj[0]);
             temp[1]=stringToNum<float>(obj[1]);
             temp[2]=stringToNum<float>(obj[2]);
             temp[3]=stringToNum<float>(obj[3]);
             //cout<<temp[0]<<","<<temp[1]<<","<<temp[2]<<","<<temp[3]<<"\n";
             pts[0][0]= temp[0];         pts[0][1]= temp[1];
             pts[1][0]= temp[0]+temp[2]; pts[1][1]= temp[1];
             pts[2][0]= temp[0]+temp[2]; pts[2][1]= temp[1]+temp[3];
             pts[3][0]= temp[0];         pts[3][1]= temp[1]+temp[3];
             for(int i=0;i<4;++i){
                str_pts+="[";
                for(int j=0;j<2;++j){
                  str_pts+=to_string(pts[i][j]);
                if(j==0) str_pts+=",";
              }
              if(i!=3)
                str_pts+="],";
              else
                str_pts+="]";
             }
             str_pts+="]}";
           }
           vec_pts.push_back(str_pts);
           fin_rect.close();
        }
        file.close();

        inference::InferenceRequests inference_requests;
        auto *im_fp = fopen(str_img.c_str(),"rb");
        fseek(im_fp, 0, SEEK_END);
        auto im_size = ftell(im_fp);
        rewind(im_fp);
        std::vector<char> image_data(im_size, 0);
        fread(image_data.data(),1,im_size, im_fp);
        fclose(im_fp);
        auto *request = inference_requests.add_requests();
        request->mutable_data()->set_uri(str_img);
        request->mutable_data()->set_body(image_data.data(), im_size);
        string jsoninput=R"({"detections":[)";
        for(int i=0;i<vec_pts.size();i++){
          
          string str_pts=vec_pts[i];
          jsoninput+=str_pts;
        }
        jsoninput+="]}";
        cout<<jsoninput<<endl;
        request->mutable_data()->set_attribute(jsoninput);
        auto inference_requests_size = inference_requests.ByteSize();
        std::vector<char> inference_requests_data(inference_requests_size, 0);
        inference_requests.SerializeToArray(inference_requests_data.data(),
                                              inference_requests_size);

        std::vector<char> inference_responses_data(4 * 1024 * 1024, 0);
        Tron::Timer timer;
        double time_cost = 0;
        timer.start();
        int inference_responses_size;
        netInference(handle, inference_requests_data.data(),
                       inference_requests_size, &code, &err,
                       inference_responses_data.data(), &inference_responses_size);
        time_cost += timer.get_millisecond();
        std::cout << "Time cost: " << time_cost << " ms.\n";

        if (code == 200) {
          inference::InferenceResponses inference_responses;
          inference_responses.ParseFromArray(inference_responses_data.data(),
                                             inference_responses_size);

          std::cout << inference_responses.responses(0).result() << std::endl;
        } else {
          std::cout << err << std::endl;
        }
      }
    fin.close();
   }//if (code == 200)
  return 0;
}


/*
       cv::Mat im_mat=cv::imread(im_file);
       for(int k=0;k<136;k+=2)
           cv::circle(im_mat,cv::Point(face_attribute[0][k+3],face_attribute[0][k+1+3]),4,cv::Scalar(255,0,0,0),8);
       std::string im_file_dep=im_deploy_list[i];
       cv::imwrite(im_file_dep.c_str(),im_mat);
       //cv::resize(im_mat,im_mat,cv::Size(im_mat.cols/3,im_mat.rows/3));
       //cv::imshow("im_mat",im_mat);
       //cv::waitKey();
       //}
*/

