#ifndef __NET_HPP__
#define __NET_HPP__

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;

namespace Shadow {

enum ShadowStatus {
  shadow_status_success = 200,
};
enum InterMethod {
  nearest = 0,
  bilinear = 1,
};

class Net {
 public:
  /*
      单模型初始化模型函数。可支持单个模型的初始化，后面将移除：
      -- gpuID 指定GPU的ID
      -- data  模型engine的数据
      -- size  模型engine的大小
    */
  //virtual ShadowStatus init(const int gpuID, void *data, const int size) = 0;

  /*
      多模型初始化模型函数。可支持单个或多个模型的初始化，推荐此用法：
      -- gpuID 指定GPU的ID
      -- data  模型engine的数据，每一个engine用vector<char>存储
      -- size  模型engine的大小，每一个engine用int存储
    */
  virtual ShadowStatus init(const int gpuID, const std::vector<std::vector<char>> data, const std::vector<int> size) = 0;

  /*
      end-to-end无感知推理函数，推荐此用法：
      -- imgs opencv解码后的图像batch
      -- attributes 每张图像的属性，以json的格式给出，例如要对图像中指定的目标进行后续处理，此处要给出对应的检测框坐标json，可为空
      -- results 返回json格式的结果
    */
  virtual ShadowStatus predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &attributes, std::vector<std::string> &results) = 0;

  /*
      返回网络输出的单引擎推理函数，即不包含后处理：
      -- imgs opencv解码后的图像batch
      -- outputlayer 输出层的名字
      -- results 网络输出层的结果
      -- enginIndex 推理第enginIndex个网络，默认第0个
    */
  virtual ShadowStatus predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &outputlayer, std::vector<std::vector<float>> &results, int enginIndex = 0) = 0;

  /*
     返回网络输出的单引擎推理函数，即不包含前处理和后处理：
     -- imgs 前处理后的数据
     -- outputlayer 输出层的名字
     -- results 网络输出层的结果
     -- enginIndex 推理第enginIndex个网络，默认第0个
     */
  virtual ShadowStatus predict(const std::vector<std::vector<float>> &imgs, const std::vector<std::string> &outputlayer, std::vector<std::vector<float>> &results, int enginIndex = 0) = 0;

  /*
      析构函数，释放内存
    */
  virtual ShadowStatus destroy() = 0;
};

/*
  end-to-end单模型无感知创建网络函数，考虑到兼容，后面将移处：
  --inputShape [channel，width，height]，每一个模型preParam的size为3,分别对应网络输入的通道、长与宽
  --preParam mean及scale，每一个模型preParam的size为4或6，分别对应于[b_meam,g_mean,r_mean,scale]或[b_meam,g_mean,r_mean,b_scale,g_scale,r_scale]
  --method 插值方式，默认bilinear
*/
//Net *createNet(int batchSize, const int *inputShape, float *preParam, InterMethod method = bilinear);

/*
  end-to-end单或多模型无感知创建网络函数，推荐此用法：
  --modelNum 模型的个数
  --method 插值方式，默认bilinear
*/
Net *createNet(int modelNum, InterMethod method = bilinear);

/*
  单或多模型创建网络函数，即自定义归一化参数：
  --inputShape [channel，width，height]，每一个模型preParam的size为3,分别对应网络输入的通道、长与宽
  --preParam mean及scale，每一个模型preParam的size为4或6，分别对应于[b_meam,g_mean,r_mean,scale]或[b_meam,g_mean,r_mean,b_scale,g_scale,r_scale]
  --method 插值方式，默认bilinear
*/
Net *createNet(vector<vector<int>> &inputShape, vector<vector<float>> &preParam, InterMethod method = bilinear);

}  // namespace Shadow
#endif /* Net_hpp */
