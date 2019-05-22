
project(framework CXX)

find_package(cppzmq REQUIRED QUIET)

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fPIC")
else ()
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /wd4005 /wd4018 /wd4244 /wd4267 /wd4291 /wd4305 /wd4309 /wd4800 /wd4819 /wd4996")
endif ()

file(GLOB lib_srcs *.cpp *.hpp)
list(FILTER lib_srcs EXCLUDE REGEX "_test|_main")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_library(tron_framework ${lib_srcs})
else ()
  add_library(tron_framework STATIC ${lib_srcs})
endif ()

target_link_libraries(tron_framework
                      PRIVATE cppzmq
                      PUBLIC zmq
                      PUBLIC glog
                      tron_common tron_proto
                      ${Tron_LINKER_LIBS})
target_include_directories(tron_framework
                           PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(framework_example example_main.cpp ${lib_srcs})
target_link_libraries(framework_example PRIVATE
                         PUBLIC zmq
                         tron_common)

add_test(NAME framework_test COMMAND framework_test)
add_executable(framework_test example_test.cpp ${lib_srcs})
target_link_libraries(framework_test gtest_main glog benchmark
                      tron_common)
