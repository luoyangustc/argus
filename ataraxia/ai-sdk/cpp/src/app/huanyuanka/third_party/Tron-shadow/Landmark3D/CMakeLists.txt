cmake_minimum_required(VERSION 3.2)
set(PROJECT_NAME landmark)
project(${PROJECT_NAME})

option(USE_OpenCV "Use OpenCV to read, write and show image" ON)
option(USE_OpenCV_GPU "Use GPU OpenCV to read, write and show image" ON)
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
include_directories(SYSTEM ${CUDA_TOOLKIT_INCLUDE})

include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/TensorRT-4.0.1.6/include ${PROJECT_SOURCE_DIR}/../3rdparty/rapidJson ${PROJECT_SOURCE_DIR}/../common)
link_directories(${PROJECT_SOURCE_DIR}/../3rdparty/TensorRT-4.0.1.6/lib)

set(${PROJECT_NAME}_lib_src)
set(${PROJECT_NAME}_lib_common)
set(${PROJECT_NAME}_examples_src)

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(../common common)

cuda_add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_lib_src} ${${PROJECT_NAME}_lib_common})
target_link_libraries(${PROJECT_NAME} ${${PROJECT_NAME}_LINKER_LIBS} ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY} nvinfer_plugin cudnn nvinfer nvparsers)

foreach (${PROJECT_NAME}_example_src ${${PROJECT_NAME}_examples_src})
    message(STATUS ${${PROJECT_NAME}_example_src})
	get_filename_component(example_fil_we ${${PROJECT_NAME}_example_src} NAME_WE)
    include_directories(${PROJECT_SOURCE_DIR}/src)
    cuda_add_executable(${example_fil_we} ${${PROJECT_NAME}_example_src})
	target_link_libraries(${example_fil_we} ${PROJECT_NAME} ${${PROJECT_NAME}_LINKER_LIBS})
endforeach ()


#file(GLOB tensorrt_src *.cpp *.cu)
#cuda_add_executable(test_tensorrt ${tensorrt_src})
option(USE_OpenCV "Use OpenCV to read, write and show image" ON)
#target_link_libraries(teGst_tensorrt ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY} cudnn nvinfer nvparsers)
