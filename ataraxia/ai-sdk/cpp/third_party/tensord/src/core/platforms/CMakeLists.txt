
project(platforms CXX)

add_subdirectory(foo)
if (${PLATFORM_MXNET} OR ${PLATFORM_ALL})
  add_subdirectory(mxnet)
endif ()
if (${PLATFORM_CAFFE} OR ${PLATFORM_ALL})
  add_subdirectory(caffe)
endif ()

set(TENSORD_PLATFORMS_LINK ${TENSORD_PLATFORMS_LINK} PARENT_SCOPE)
set(TENSORD_PLATFORMS_DEFINITION ${TENSORD_PLATFORMS_DEFINITION} PARENT_SCOPE)
