cmake_minimum_required(VERSION 2.8.12)

project(eigen-download NONE)

include(ExternalProject)

ExternalProject_Add(eigen
                    URL "http://oxmz2ax9v.bkt.clouddn.com/eigen3.tar"
                    URL_MD5 5540191f86cc26777f45781fb1fb5ed8
                    DOWNLOAD_NO_PROGRESS TRUE
                    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third_party"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/eigen3"
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
                    )
