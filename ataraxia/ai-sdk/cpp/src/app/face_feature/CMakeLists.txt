cmake_minimum_required(VERSION 3.2)
project(face_feature)

option(USE_CUDA "Use CUDA" ON)
option(USE_CUDNN "Use CUDNN for acceleration" ON)

option(USE_OpenCV "Use OpenCV to read, write and show image" ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)

set(OpenCV_DIR "/usr/local/opencv-3.4.2" CACHE PATH "OpenCV root directory")

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
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -I/usr/local/opt/openblas/include -fsanitize=address -o1")
else ()
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4005 /wd4018 /wd4244 /wd4267 /wd4291 /wd4305 /wd4309 /wd4800 /wd4819 /wd4996")
endif ()

set(PLATFORM_CAFFE ON CACHE BOOL "")
set(PLATFORM_MXNET ON CACHE BOOL "")
add_subdirectory(${AISDK_SOURCE_DIR}/third_party/tensord
                 ${PROJECT_BINARY_DIR}/third_party/tensord)
include_directories(${AISDK_SOURCE_DIR}/third_party/tensord/include)

include_directories(".")

file(GLOB src_ff "*.cpp" "*.hpp" "**/*.cpp" "**/*.hpp")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_library(inference_ff OBJECT ${src_ff})
  add_library(test_ff OBJECT test_tron.cpp)
else ()
  add_library(inference_ff ${src_ff})
  add_executable(test_ff test_tron.cpp)
endif ()

target_link_libraries(inference_ff
                      PUBLIC protobuf
                      PUBLIC caffe
                      PUBLIC mxnet
                      PUBLIC glog
                      PUBLIC boost_system
                      PUBLIC tensord_core
                      tron_common tron_proto
                      ${Tron_LINKER_LIBS})
install(TARGETS inference_ff DESTINATION ${Tron_INSTALL_LIB_PREFIX})

target_link_libraries(test_ff
                      PUBLIC protobuf
                      PUBLIC caffe
                      PUBLIC mxnet
                      PUBLIC glog
                      PUBLIC boost_system
                      PUBLIC tensord_core
                      inference_ff
                      tron_common tron_proto
                      ${Tron_LINKER_LIBS})
install(TARGETS test_ff DESTINATION ${Tron_INSTALL_BIN_PREFIX})

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
include(CPack)
