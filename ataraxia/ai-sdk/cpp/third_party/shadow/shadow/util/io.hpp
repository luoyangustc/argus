#ifndef SHADOW_UTIL_IO_HPP
#define SHADOW_UTIL_IO_HPP

#if defined(USE_Protobuf)
#include <google/protobuf/message.h>

#if GOOGLE_PROTOBUF_VERSION >= 3003000
#define SUPPORT_JSON
#endif

#else
#include "core/params.hpp"
#endif

namespace Shadow {

namespace IO {

#if defined(USE_Protobuf)
using google::protobuf::Message;

bool ReadProtoFromText(const std::string& proto_text, Message* proto);

bool ReadProtoFromTextFile(const std::string& proto_file, Message* proto);

bool ReadProtoFromBinaryFile(const std::string& proto_file, Message* proto);

void WriteProtoToText(const Message& proto, std::string* proto_text);

void WriteProtoToTextFile(const Message& proto, const std::string& proto_file);

void WriteProtoToBinaryFile(const Message& proto,
                            const std::string& proto_file);

#if defined(SUPPORT_JSON)
void WriteProtoToJsonText(const Message& proto, std::string* json_text,
                          bool compact = false);
#endif

#else
bool ReadProtoFromText(const std::string& proto_text, tron::NetParam* proto);

bool ReadProtoFromTextFile(const std::string& proto_file,
                           tron::NetParam* proto);
#endif

}  // namespace IO

}  // namespace Shadow

#endif  // SHADOW_UTIL_IO_HPP
