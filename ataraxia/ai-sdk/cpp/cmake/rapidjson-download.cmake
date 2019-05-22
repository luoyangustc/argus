cmake_minimum_required(VERSION 2.8.12)

project(rapidjson-download NONE)

include(ExternalProject)

ExternalProject_Add(rapidjson
        SOURCE_DIR "@_BUILD_ROOT@/rapidjson-src"
        BINARY_DIR "@_BUILD_ROOT@/rapidjson-build"
        GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
        GIT_TAG v1.1.0
        CMAKE_ARGS "-DRAPIDJSON_BUILD_DOC=OFF"
                   "-DRAPIDJSON_BUILD_EXAMPLES=OFF"
                   "-DRAPIDJSON_BUILD_TESTS=OFF"
                   "-DCMAKE_INSTALL_PREFIX:PATH=@_PROJECT_ROOT@/third_party/rapidjson"
)
