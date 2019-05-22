cmake_minimum_required(VERSION 2.8.12)

project(gtest-download NONE)

# Download and install GoogleTest
include(ExternalProject)
ExternalProject_Add(gtest
    URL https://github.com/google/googletest/archive/release-1.8.0.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third_party"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/gtest"
)
ExternalProject_Get_Property(gtest source_dir binary_dir)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

include_directories("${source_dir}/googletest/include"
                    "${source_dir}/googlemock/include")