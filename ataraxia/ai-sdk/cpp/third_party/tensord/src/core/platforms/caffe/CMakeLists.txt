
if (${TENSORD_DEV})
    include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_definitions(-DCPU_ONLY)
    add_library(platform_caffe OBJECT
                caffe.cc caffe.hpp)
else ()
    add_library(platform_caffe
                caffe.cc caffe.hpp)
endif ()
if (${USE_CUDA})
    include_directories(SYSTEM ${CUDA_TOOLKIT_INCLUDE})
    target_link_libraries(platform_caffe
                          PRIVATE caffe
                          PRIVATE protobuf
                          PRIVATE glog
                          PRIVATE proto
                          ${CUDA_CUDART_LIBRARY}
                          ${CUDA_cublas_LIBRARY})
else ()
    target_link_libraries(platform_caffe
                          PRIVATE caffe
                          PRIVATE protobuf
                          PRIVATE glog
                          PRIVATE proto)
endif ()
install(TARGETS platform_caffe DESTINATION lib)
set(TENSORD_PLATFORMS_LINK ${TENSORD_PLATFORMS_LINK} platform_caffe PARENT_SCOPE)
set(TENSORD_PLATFORMS_DEFINITION ${TENSORD_PLATFORMS_DEFINITION} PLATFORM_CAFFE PARENT_SCOPE)
