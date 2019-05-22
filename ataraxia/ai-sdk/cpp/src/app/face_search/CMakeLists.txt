cmake_minimum_required(VERSION 3.2)
project(face_search)

set(CMAKE_FIND_ROOT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
if (NOT CMAKE_INSTALL_PREFIX)
  message(STATUS "CMake install prefix not set - defaulting to ${CMAKE_BINARY_DIR}/distribute")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/distribute)
endif ()

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Release Debug." FORCE)
endif ()

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -I/usr/local/opt/openblas/include")
else ()
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4005 /wd4018 /wd4244 /wd4267 /wd4291 /wd4305 /wd4309 /wd4800 /wd4819 /wd4996")
endif ()

include_directories(".")

set(src_fs infer.cpp infer.hpp)

add_library(inference_fs ${src_fs})
target_link_libraries(inference_fs
                      PUBLIC protobuf
                      PUBLIC glog
                      tron_common tron_proto
                      ${Tron_LINKER_LIBS})
install(TARGETS inference_fs DESTINATION ${Tron_INSTALL_LIB_PREFIX})

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
include(CPack)