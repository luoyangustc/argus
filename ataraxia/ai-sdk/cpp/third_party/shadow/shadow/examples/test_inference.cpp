#include "core/network.hpp"
#include "util/io.hpp"
#include "util/util.hpp"

#include <opencv2/opencv.hpp>

using namespace Shadow;

static inline void ConvertData(const cv::Mat &im_mat, float *data, int channel,
                               int height, int width, int flag = 1,
                               bool transpose = false) {
  CHECK(!im_mat.empty());
  CHECK_NOTNULL(data);

  int c_ = im_mat.channels(), h_ = im_mat.rows, w_ = im_mat.cols;
  int dst_spatial_dim = height * width;

  float *data_r = nullptr, *data_g = nullptr, *data_b = nullptr,
        *data_gray = nullptr;
  if (channel == 3 && flag == 0) {
    // Convert to RRRGGGBBB
    data_r = data;
    data_g = data + dst_spatial_dim;
    data_b = data + (dst_spatial_dim << 1);
  } else if (channel == 3 && flag == 1) {
    // Convert to BBBGGGRRR
    data_r = data + (dst_spatial_dim << 1);
    data_g = data + dst_spatial_dim;
    data_b = data;
  } else if (channel == 1) {
    // Convert to Gray
    data_gray = data;
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }

  cv::Size cv_size(width, height);

  cv::Mat im_resize;
  cv::resize(im_mat, im_resize, cv_size);

  int dst_h = height, dst_w = width;
  if (transpose) {
    cv::transpose(im_resize, im_resize);
    dst_h = width, dst_w = height;
  }

  if (channel == 3) {
    CHECK_EQ(c_, 3);
    for (int h = 0; h < dst_h; ++h) {
      const auto *data_src = im_resize.ptr<uchar>(h);
      for (int w = 0; w < dst_w; ++w) {
        *data_b++ = static_cast<float>(*data_src++);
        *data_g++ = static_cast<float>(*data_src++);
        *data_r++ = static_cast<float>(*data_src++);
      }
    }
  } else if (channel == 1) {
    cv::Mat im_gray;
    cv::cvtColor(im_resize, im_gray, cv::COLOR_BGR2GRAY);
    for (int h = 0; h < dst_h; ++h) {
      const auto *data_src = im_gray.ptr<uchar>(h);
      for (int w = 0; w < dst_w; ++w) {
        *data_gray++ = static_cast<float>(*data_src++);
      }
    }
  } else {
    LOG(FATAL) << "Unsupported flag " << flag;
  }
}

int main(int argc, char const *argv[]) {
  std::string model_file("models/classify/squeezenet_v1.1_merged.tronmodel");
  std::string im_file("data/static/cat.jpg");

  /* Read protobuf from binary file */
  tron::MetaNetParam meta_net_param;
  if (!IO::ReadProtoFromBinaryFile(model_file, &meta_net_param)) {
    LOG(FATAL) << "Load model " << model_file << " error";
  }

  /* Set up gpu id and load model */
  Network network;
  network.Setup(0);
  network.LoadModel(meta_net_param.network(0));

  /* Get input, output blob names and input data shape */
  const auto &in_blob = network.in_blob();
  const auto &out_blob = network.out_blob();
  const auto &data_shape = network.GetBlobByName<float>(in_blob[0])->shape();

  /* Read image and convert to network's input data */
  const auto &im_mat = cv::imread(im_file);
  int num_data = data_shape[0] * data_shape[1] * data_shape[2] * data_shape[3];
  std::vector<float> input_data(num_data, 0);
  ConvertData(im_mat, input_data.data(), data_shape[1], data_shape[2],
              data_shape[3]);

  /* Prepare data map for network forward */
  std::map<std::string, float *> data_map;
  data_map[in_blob[0]] = input_data.data();

  network.Forward(data_map);

  /* Get forward output data to vector float */
  const auto *softmax_data = network.GetBlobDataByName<float>(out_blob[0]);
  int num_prob = network.GetBlobByName<float>(out_blob[0])->num();
  std::vector<float> task_score(softmax_data, softmax_data + num_prob);

  /* Print top k results */
  const auto &top_index = Util::top_k(task_score, 3);
  for (const auto index : top_index) {
    std::cout << index << ": " << task_score[index] << std::endl;
  }

  /* Release network */
  network.Release();

  return 0;
}
