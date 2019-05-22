#ifndef TRON_COMMON_PROTOBUF_HPP  // NOLINT
#define TRON_COMMON_PROTOBUF_HPP

#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/text_format.h"

namespace tron {

static inline bool read_proto_from_array(const void *proto_data, int proto_size,
                                         google::protobuf::Message *proto) {
  return proto->ParseFromArray(proto_data, proto_size);
  //   using google::protobuf::io::ArrayInputStream;
  //   using google::protobuf::io::CodedInputStream;
  //   auto *param_array_input = new ArrayInputStream(proto_data, proto_size);
  //   auto *param_coded_input = new CodedInputStream(param_array_input);
  //   param_coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
  //   bool success = proto->ParseFromCodedStream(param_coded_input) &&
  //                  param_coded_input->ConsumedEntireMessage();
  //   delete param_coded_input;
  //   delete param_array_input;
  //   return success;
}

}  // namespace tron

#endif  // TRON_COMMON_PROTOBUF_HPP NOLINT