cmake_minimum_required(VERSION 3.2)
project(foo)

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
else ()
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4005 /wd4018 /wd4244 /wd4267 /wd4291 /wd4305 /wd4309 /wd4800 /wd4819 /wd4996")
endif ()

include_directories(".")

file(GLOB proto_files "*.proto")
GenProtobuf(proto_srcs proto_hdrs ${proto_files})
add_library(proto_foo STATIC ${proto_srcs} ${proto_hdrs})

set(src_foo infer.cpp infer.hpp
            inference.cpp inference.hpp
            forward.cpp forward.hpp
            ${proto_srcs} ${proto_hdrs})

add_library(inference_foo ${src_foo})
target_link_libraries(inference_foo
                      PUBLIC zmq
                      PUBLIC protobuf
                      proto_foo
                      tron_common tron_proto
                      ${Tron_LINKER_LIBS})
install(TARGETS inference_foo DESTINATION ${Tron_INSTALL_LIB_PREFIX})

add_executable(test_foo test_tron.cpp)
target_link_libraries(test_foo
                      PUBLIC zmq
                      PUBLIC protobuf
                      PUBLIC boost_system
                      inference_foo tron_common tron_proto
                      ${Tron_LINKER_LIBS})
install(TARGETS test_foo DESTINATION ${Tron_INSTALL_BIN_PREFIX})

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
include(CPack)