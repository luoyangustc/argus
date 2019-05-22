include(FindPackageHandleStandardArgs)

set(TensorRT_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA TensorRT")

set(TensorRT_DIR ${TensorRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} /usr /usr/local)

find_path(TensorRT_INCLUDE_DIRS
          NAMES NvInfer.h
          PATHS ${TensorRT_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "TensorRT include header NvInfer.h")

find_library(TensorRT_NVINFER_LIBRARY
             NAMES nvinfer
             PATHS ${TensorRT_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "TensorRT library")

find_library(TensorRT_NVINFER_PLUGIN_LIBRARY
             NAMES nvinfer_plugin
             PATHS ${TensorRT_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "TensorRT library")

find_library(TensorRT_NVPARSERS_LIBRARY
             NAMES nvparsers
             PATHS ${TensorRT_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
             DOC "TensorRT library")

set(TensorRT_LIBRARIES ${TensorRT_NVINFER_LIBRARY} ${TensorRT_NVINFER_PLUGIN_LIBRARY} ${TensorRT_NVPARSERS_LIBRARY})

find_package_handle_standard_args(TensorRT DEFAULT_MSG TensorRT_INCLUDE_DIRS TensorRT_LIBRARIES)

if (TensorRT_FOUND)
  parse_header(${TensorRT_INCLUDE_DIRS}/NvInfer.h
               NV_TENSORRT_MAJOR NV_TENSORRT_MINOR NV_TENSORRT_PATCH)
  if (NOT NV_TENSORRT_MAJOR)
    set(TensorRT_VERSION "?")
  else ()
    set(TensorRT_VERSION "${NV_TENSORRT_MAJOR}.${NV_TENSORRT_MINOR}.${NV_TENSORRT_PATCH}")
  endif ()
  if (NOT TensorRT_FIND_QUIETLY)
    message(STATUS "Found TensorRT: ${TensorRT_INCLUDE_DIRS}, ${TensorRT_LIBRARIES} (found version ${TensorRT_VERSION})")
  endif ()
  mark_as_advanced(TensorRT_ROOT_DIR TensorRT_INCLUDE_DIRS TensorRT_LIBRARIES)
else ()
  if (TensorRT_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find TensorRT")
  endif ()
endif ()
