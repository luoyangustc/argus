
#include "tensord/core/net_list.hpp"

#include "glog/logging.h"

#include "core/platforms/foo/foo.hpp"
#ifdef PLATFORM_CAFFE
#include "core/platforms/caffe/caffe.hpp"
#endif
#ifdef PLATFORM_MXNET
#include "core/platforms/mxnet/mxnet.hpp"
#endif

namespace tensord {
namespace core {

static std::map<std::string, NewNetFunc> PLATFORMS = {
    {"foo", platform::Foo::Create},
#ifdef PLATFORM_CAFFE
    {"caffe", platform::Caffe::Create},
#endif
#ifdef PLATFORM_MXNET
    {"mxnet", platform::Mxnet::Create},
#endif
};

std::map<std::string, NewNetFunc> AllPlatforms() {
  return PLATFORMS;
}

void RegisterPlatform(NewNetFunc func, std::string name) {
  PLATFORMS[name] = func;
}

}  // namespace core
}  // namespace tensord
