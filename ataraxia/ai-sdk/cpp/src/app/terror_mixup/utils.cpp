#include "./utils.hpp"
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <boost/algorithm/string.hpp>

namespace tron {
namespace terror_mixup {

using std::string;
vector<string> split_str(const string &s, const string &seq) {
  vector<string> fields;
  boost::algorithm::split(fields, s, boost::algorithm::is_any_of(seq),
                          boost::algorithm::token_compress_on);
  return fields;
}

csv_fields csv_parse(const string &csv_content, string delimiter) {
  csv_fields csv_fields;
  for (auto &line : split_str(csv_content, "\n\r")) {
    auto fields = split_str(line, delimiter);
    if (!fields.empty() && !fields[0].empty() && !isspace(fields[0][0])) {
      csv_fields.push_back(fields);
    }
  }
  if (!csv_fields.empty()) {
    for (auto &line_fields : csv_fields) {
      if (line_fields.size() != csv_fields[0].size()) {
        throw std::invalid_argument("bad csv, number of columns not matching");
      }
    }
  }
  return csv_fields;
}

string read_bin_file_to_string(const string &filename) {
  std::ifstream in(filename, std::iostream::in | std::iostream::binary);
  if (!in) {
    throw std::invalid_argument("open file failed");
  }
  in.seekg(0, std::istream::end);
  int file_size = in.tellg();
  if (file_size == -1) {
    throw std::invalid_argument("seek file failed");
  }
  in.seekg(0, std::istream::beg);
  string result;
  result.resize(file_size);
  in.read(&result[0], file_size);
  return result;
}

string dump_msg(const google::protobuf::Message &message) {
  string out;
  auto option = google::protobuf::util::JsonOptions();
  option.always_print_primitive_fields = true;
  auto r = google::protobuf::util::MessageToJsonString(message, &out, option);
  CHECK(r.ok() == true) << "json dump failed";
  return out;
}

string diff_protobuf_msg_with_precision(
    const google::protobuf::Message &message1,
    const google::protobuf::Message &message2, float precision) {
  google::protobuf::util::MessageDifferencer differ;
  {
    google::protobuf::util::DefaultFieldComparator field_comparitor;
    field_comparitor.set_float_comparison(
        google::protobuf::util::DefaultFieldComparator::APPROXIMATE);
    field_comparitor.SetDefaultFractionAndMargin(0, precision);
    differ.set_field_comparator(&field_comparitor);
  }
  {
    string rs;
    differ.ReportDifferencesToString(&rs);
    auto r2 = differ.Compare(message1, message2);
    if (r2) {
      CHECK_EQ(rs, "");
    } else {
      CHECK_NE(rs, "");
    }
    return rs;
  }
}
void vector_marshal_to_string(const vector<float> &in, string &out) {
  out.resize(in.size() * sizeof(float));
  memcpy(&out[0], in.data(), in.size() * sizeof(float));
}

void string_unmarshal_to_vector(const string &in, vector<float> &out) {
  CHECK_EQ(in.size() % sizeof(float), 0);
  out.resize(in.size() / sizeof(float));
  memcpy(out.data(), in.data(), in.size());
}

string vector_char_to_string(const vector<char> &buf) {
  return string(reinterpret_cast<const char *>(&buf[0]), buf.size());
}

vector<float> join_batch_size_data(const std::vector<vector<float>> &args) {
  CHECK_GT(args.size(), 0);
  int item_size = args.begin()->size();
  vector<float> r(item_size * args.size());
  int i = 0;
  for (auto &arg : args) {
    CHECK_EQ(arg.size(), item_size);
    std::copy(arg.begin(), arg.end(), r.begin() + item_size * i);
    i++;
  }
  return r;
}
}  // namespace terror_mixup
}  // namespace tron
