cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(cppzmq-download NONE)

include(ExternalProject)

set(_CMAKE_ARGS "-DCPPZMQ_BUILD_TESTS=OFF"
                "-DCMAKE_INSTALL_PREFIX:PATH=@_PROJECT_ROOT@/third_party/cppzmq")
if (APPLE)
    list(APPEND ${_CMAKE_ARGS} "-DENABLE_DRAFTS=OFF")
endif ()

ExternalProject_Add(cppzmq
    SOURCE_DIR      "@_BUILD_ROOT@/cppzmq-src"
    BINARY_DIR      "@_BUILD_ROOT@/cppzmq-build"
    GIT_REPOSITORY  https://github.com/zeromq/cppzmq.git
    GIT_TAG         v4.3.0
    CMAKE_ARGS      ${_CMAKE_ARGS}
)

unset(_CMAKE_ARGS)
