cmake_minimum_required(VERSION 3.2)
set(PROJECT_NAME mix)
project(${PROJECT_NAME})

option(USE_OpenCV "Use OpenCV to read, write and show image" ON)
option(USE_OpenCV_GPU "Use GPU OpenCV to read, write and show image" OFF)
set(${PROJECT_NAME}_LINKER_LIBS)

if (${USE_OpenCV})
  find_package(OpenCV PATHS ${OpenCV_DIR} NO_DEFAULT_PATH QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
  if (NOT OpenCV_FOUND) # if not OpenCV 3.x, then try to find OpenCV 2.x in default path
    find_package(OpenCV REQUIRED QUIET COMPONENTS core highgui imgproc)
  endif ()
  if (${USE_OpenCV_GPU}) # if using opencv GPU add the GPU opencvlibs 
    find_package(OpenCV REQUIRED QUIET COMPONENTS cudaimgproc cudawarping)
  endif ()
  if (${OpenCV_VERSION} VERSION_GREATER "2.4.13")
    find_package(OpenCV REQUIRED QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
  endif ()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND ${PROJECT_NAME}_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "Found OpenCV: ${OpenCV_CONFIG_PATH} (found version ${OpenCV_VERSION})")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

set(CUDA_PROPAGATE_HOST_FLAGS ON)
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=\"compute_30,sm_61\"")

find_package(CUDA QUIET)
if (CUDA_FOUND)
  include_directories(SYSTEM ${CUDA_TOOLKIT_INCLUDE})
else ()
  set(CUDA_CUDART_LIBRARY "")
  set(CUDA_TOOLKIT_INCLUDE "")
  include_directories(${PROJECT_SOURCE_DIR}/include/cuda)
endif ()

# include_directories(${PROJECT_SOURCE_DIR}/include/TensorRT)
# link_directories(${PROJECT_SOURCE_DIR}/../3rdparty/TensorRT-4.0.1.6/lib)

set(${PROJECT_NAME}_lib_common)
add_subdirectory(third_party/Tron-shadow/common)
include_directories(".")
include_directories(third_party/Tron-shadow/common)

add_subdirectory(${AISDK_SOURCE_DIR}/third_party/tensord
                 ${PROJECT_BINARY_DIR}/third_party/tensord)
include_directories(${AISDK_SOURCE_DIR}/third_party/tensord/include)

file(GLOB ${PROJECT_NAME}_src *.cc *.cpp *.hpp *.cu)

add_definitions(-Wno-pedantic)
if (CUDA_FOUND)
  cuda_add_library(inference_${PROJECT_NAME} SHARED
                   ${${PROJECT_NAME}_src}
                   ${${PROJECT_NAME}_lib_common})
  target_link_libraries(inference_${PROJECT_NAME}
                        protobuf
                        glog
                        ${${PROJECT_NAME}_LINKER_LIBS}
                        ${CUDA_CUDART_LIBRARY}
                        ${CUDA_cublas_LIBRARY}
                        nvinfer_plugin
                        cudnn
                        nvinfer
                        nvparsers
                        tensord_core
                        tron_common tron_proto
                        ${Tron_LINKER_LIBS})
else ()
  cuda_add_library(inference_${PROJECT_NAME} OBJECT
                   ${${PROJECT_NAME}_src}
                   ${${PROJECT_NAME}_lib_common})
endif ()