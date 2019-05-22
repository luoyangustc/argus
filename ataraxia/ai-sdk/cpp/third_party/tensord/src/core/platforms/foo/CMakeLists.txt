
add_library(platform_foo foo.cc foo.hpp)
target_link_libraries(platform_foo
                      PRIVATE protobuf
                      PRIVATE glog
                      PRIVATE proto)
install(TARGETS platform_foo DESTINATION lib)
set(TENSORD_PLATFORMS_LINK ${TENSORD_PLATFORMS_LINK} platform_foo PARENT_SCOPE)
set(TENSORD_PLATFORMS_DEFINITION ${TENSORD_PLATFORMS_DEFINITION} PLATFORM_FOO PARENT_SCOPE)
