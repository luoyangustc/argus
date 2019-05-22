
if (${TENSORD_DEV})
    include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_library(platform_mxnet OBJECT
                mxnet.cc mxnet.hpp)
else ()
    add_library(platform_mxnet
                mxnet.cc mxnet.hpp)
endif ()
target_link_libraries(platform_mxnet
                      PRIVATE mxnet
                      PRIVATE protobuf
                      PRIVATE glog
                      PRIVATE proto)
install(TARGETS platform_mxnet DESTINATION lib)
set(TENSORD_PLATFORMS_LINK ${TENSORD_PLATFORMS_LINK} platform_mxnet PARENT_SCOPE)
set(TENSORD_PLATFORMS_DEFINITION ${TENSORD_PLATFORMS_DEFINITION} PLATFORM_MXNET PARENT_SCOPE)
