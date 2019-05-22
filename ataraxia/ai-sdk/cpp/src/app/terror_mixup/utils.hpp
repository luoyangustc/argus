#ifndef TRON_TERROR_MIXUP_UTILS
#define TRON_TERROR_MIXUP_UTILS

#include <glog/logging.h>
#include <google/protobuf/message.h>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include "pprint11.h"

namespace tron {
namespace terror_mixup {

using std::string;
using std::vector;

enum class NetworkShapeType {
  Normal,
  BatchSizeUnrelated  // blob大小和batch_size无关，比如检测网络的输出
};

// 一个检查网络输入输出向量尺寸是否正确的工具类
class NetworkShape {
 private:
  vector<int> shape;
  int single_shape_size;
  NetworkShapeType network_shape_type;
  string name;

 public:
  NetworkShape(string name, vector<int> shape,
               NetworkShapeType network_shape_type)
      : shape(shape), network_shape_type(network_shape_type), name(name) {
    CHECK_EQ(shape.at(0), -1);
    CHECK_GE(shape.size(), 2);
    single_shape_size = std::accumulate(shape.begin() + 1,
                                        shape.end(),
                                        1,
                                        std::multiplies<int>());
  }
  // 检查数据的shape时候正确
  // 如果是检测网络的输出，则输出大小和batch_size无关，否则为single_batch_size*batch_size
  void assert_shape_match(const vector<float> &data, int batch_size) const {
    if (network_shape_type == NetworkShapeType::BatchSizeUnrelated) {
      CHECK_EQ(single_shape_size, data.size())
          << "shape should match, size:" << data.size()
          << " shape:" << shape << " name:"
          << name << " batch_size:" << batch_size;
      return;
    }
    CHECK_EQ(batch_size * single_shape_size, data.size())
        << "shape should match, size:" << data.size()
        << " shape:" << shape << " name:"
        << name << " batch_size:" << batch_size;
  }

  int single_batch_size() const {
    return single_shape_size;
  }
};

// 字符串split，按seq里面任意字符作为分隔符
vector<string> split_str(const string &s, const string &seq);

using csv_fields = vector<vector<string>>;
// 解析csv到一个二维字符串数组，会自动去掉空行，每行列数必须一致
csv_fields csv_parse(const string &csv_content, string delimiter = ",");

// 读取文件到string，（二进制模式，不转换换行符）,空文件会抛异常
string read_bin_file_to_string(const string &filename);

// 转换一个protobuf消息为json字符串
string dump_msg(const google::protobuf::Message &message);

// 使用指定的浮点数精度比较两个protobuf消息，如果不一致，返回diff
string diff_protobuf_msg_with_precision(
    const google::protobuf::Message &message1,
    const google::protobuf::Message &message2,
    float precision = 1e-6);

void vector_marshal_to_string(const vector<float> &in, string &out);
void string_unmarshal_to_vector(const string &in, vector<float> &out);

string vector_char_to_string(const vector<char> &buf);

vector<float> join_batch_size_data(const std::vector<vector<float>> &args);
}  // namespace terror_mixup
}  // namespace tron
#endif
