
project( tensord_tests )

download_third_party(googletest)
download_third_party(benchmark)

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${TENSORD_SOURCE_DIR}/third_party/googletest
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

find_path(GTEST_INCLUDE_DIR NAMES gtest/gtest.h
          PATHS ${TENSORD_SOURCE_DIR}/third_party/googletest/googletest/include)
find_path(GMOCK_INCLUDE_DIR NAMES gmock/gmock.h
          PATHS ${TENSORD_SOURCE_DIR}/third_party/googletest/googlemock/include)
list(APPEND GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR} ${GMOCK_INCLUDE_DIR})
find_library(GTEST_LIBEARY NAMES gtest gmock
             PATHS ${CMAKE_BINARY_DIR}/googletest-build/lib)

add_subdirectory(${TENSORD_SOURCE_DIR}/third_party/benchmark
                 ${CMAKE_BINARY_DIR}/benchmark-build
                 EXCLUDE_FROM_ALL)
include_directories(${TENSORD_SOURCE_DIR}/third_party/benchmark/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(tests testmain.cc
               core_tasks.cc
               core_engine.cc
               core_engines.cc)
target_link_libraries(tests
                      PRIVATE core
                      PRIVATE gtest_main
                      PRIVATE protobuf)
add_test(NAME tests COMMAND tests
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(benchmarks benchmarkmain.cc
               bm_core_tasks.cc)
target_link_libraries(benchmarks
                      PRIVATE core
                      PRIVATE benchmark_main
                      PRIVATE protobuf)
