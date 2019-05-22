cmake_minimum_required(VERSION 2.8.12)

project(shadow-download NONE)

include(ExternalProject)

ExternalProject_Add(shadow
                    GIT_REPOSITORY "https://oauth2:zetBRpfEFWNsv9Kghze6@gitlab.qiniu.io/luanjun/shadow.git"
                    GIT_TAG "v1.1.0"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/shadow"
                    BINARY_DIR "${CMAKE_BINARY_DIR}/shadow-build"
                    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/third_party/shadow/build"
                               "-DCMAKE_BUILD_TYPE=Release"
                               "-DUSE_CUDA=${USE_CUDA}"
                               "-DUSE_CUDNN=${USE_CUDNN}"
                               "-DUSE_Eigen=ON"
                               "-DUSE_BLAS=OFF"
                               "-DUSE_NNPACK=OFF"
                               "-DUSE_Protobuf=ON"
                               "-DUSE_OpenCV=${USE_OpenCV}"
                               "-DBUILD_EXAMPLES=OFF"
                               "-DBUILD_TEST=OFF"
                               "-DBUILD_SHARED_LIBS=ON"
                    BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target install
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
)
