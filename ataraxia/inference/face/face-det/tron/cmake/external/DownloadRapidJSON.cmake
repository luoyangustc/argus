cmake_minimum_required(VERSION 2.8.12)

project(rapidjson-download NONE)

include(ExternalProject)

ExternalProject_Add(rapidjson
                    URL "http://oxmz2ax9v.bkt.clouddn.com/rapidjson.tar"
                    URL_MD5 9bd65453bff0036f308bf984cfc8b7dc
                    DOWNLOAD_NO_PROGRESS TRUE
                    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third_party"
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/rapidjson"
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    TEST_COMMAND ""
)
