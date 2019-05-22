#ifndef TRON_COMMON_JSON_HPP
#define TRON_COMMON_JSON_HPP

#include <string>

#include "glog/logging.h"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace tron {

/**
 * Helper functions for parsing json
 */
inline rapidjson::Document get_document(const std::string &json_text) {
  rapidjson::Document document;
  if (!json_text.empty()) {
    document.Parse(json_text.c_str());
  } else {
    document.Parse("{}");
  }
  return document;
}

inline bool get_bool(const rapidjson::Value &root, const std::string &name,
                     const bool &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsBool());
    return value.GetBool();
  } else {
    return def;
  }
}

inline int get_int(const rapidjson::Value &root, const std::string &name,
                   const int &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsNumber());
    return value.GetInt();
  } else {
    return def;
  }
}

inline float get_float(const rapidjson::Value &root, const std::string &name,
                       const float &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsNumber());
    return value.GetFloat();
  } else {
    return def;
  }
}

inline std::string get_string(const rapidjson::Value &root,
                              const std::string &name, const std::string &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    assert(value.IsString());
    return value.GetString();
  } else {
    return def;
  }
}

}  // namespace tron

#endif  // TRON_COMMON_JSON_HPP NOLINT
