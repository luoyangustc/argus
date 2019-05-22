cmake_minimum_required(VERSION 2.8)

project(structure_vcs)

option(USE_MOCK_VSS "Use mock vss lib" ON)
if (USE_MOCK_VSS)
  add_definitions(-DUSE_MOCK_VSS)
endif (USE_MOCK_VSS)

enable_testing()

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
else ()
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4005 /wd4018 /wd4244 /wd4267 /wd4291 /wd4305 /wd4309 /wd4800 /wd4819 /wd4996")
endif ()

set(CMAKE_FIND_ROOT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
if (NOT CMAKE_INSTALL_PREFIX)
  message(STATUS "CMake install prefix not set - defaulting to ${CMAKE_BINARY_DIR}/distribute")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/distribute)
endif ()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)