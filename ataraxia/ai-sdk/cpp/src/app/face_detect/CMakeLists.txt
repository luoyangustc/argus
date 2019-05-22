cmake_minimum_required(VERSION 3.2)
project(tron)

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
else ()
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4005 /wd4018 /wd4244 /wd4267 /wd4291 /wd4305 /wd4309 /wd4800 /wd4819 /wd4996")
endif ()

option (USE_Shadow "use shadow" true)
if (USE_Shadow)
  find_package(shadow QUIET)
  if (NOT shadow_FOUND)
    message(STATUS "Can't find shadow library, build it from source")
    configure_file("${AISDK_SOURCE_DIR}/cmake/shadow.cmake" "${CMAKE_BINARY_DIR}/build/shadow/CMakeLists.txt")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/build/shadow")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/build/shadow")
    # download_external("${PROJECT_SOURCE_DIR}/cmake/external/DownloadShadow.cmake"
    #                   "${CMAKE_BINARY_DIR}/shadow-download")
    find_package(shadow REQUIRED QUIET)
  endif ()
  include_directories(SYSTEM ${shadow_INCLUDE_DIRS})
  list(APPEND Tron_LINKER_LIBS ${shadow_LIBRARIES})
  message(STATUS "Found Shadow: ${shadow_INCLUDE_DIRS}, ${shadow_LIBRARIES}")
  install(FILES ${shadow_LIBRARIES} DESTINATION ${Tron_INSTALL_LIB_PREFIX})
endif()

add_subdirectory(${AISDK_SOURCE_DIR}/third_party/tensord
                 ${PROJECT_BINARY_DIR}/third_party/tensord)
include_directories(${AISDK_SOURCE_DIR}/third_party/tensord/include)
 
include_directories(".")

file(GLOB fd_src "*.cpp" "*.cc" "*.hpp" "**/*.cpp" "**/*.hpp")
add_library(inference_fd ${fd_src})
target_link_libraries(inference_fd
                      PUBLIC zmq
                      PUBLIC glog
                      PUBLIC tensord_core
                      tron_common tron_proto
                      ${Tron_LINKER_LIBS})
install(TARGETS inference_fd DESTINATION ${Tron_INSTALL_LIB_PREFIX})

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
include(CPack)