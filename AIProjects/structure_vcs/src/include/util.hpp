#ifndef URL_ENCODE_HPP
#define URL_ENCODE_HPP

#include <string>
#include "document.h"

namespace util {

std::string urlencode(const std::string &s);

void merge_docs(rapidjson::Value &dstObject, rapidjson::Value &srcObject, rapidjson::Document::AllocatorType &allocator);

}

#endif