cmake_minimum_required(VERSION 3.2)
project(terror_mixup)

file(GLOB proto_files "*.proto")
GenProtobuf(proto_srcs proto_hdrs ${proto_files})

find_package(OpenCV 3.4.1 REQUIRED COMPONENTS core imgproc imgcodecs)

include_directories(".")

set(terror_mixup_inference_test_src 
    post_process.cpp post_process_test.cpp post_process.hpp
    pre_process.cpp pre_process_test.cpp pre_process.hpp
    utils_test.cpp utils.cpp utils.hpp
    debug_print.hpp
    main_test.cpp
    ${proto_srcs} ${proto_hdrs}
    )

set(terror_mixup_forward_test_src
    ${terror_mixup_inference_test_src}
    process_test.cpp
    forward.hpp
    inference.hpp
    infer.cpp infer.hpp
    process.cpp process.hpp
)

set(inference_terror_mixup_src ${terror_mixup_forward_test_src})
list(FILTER inference_terror_mixup_src EXCLUDE REGEX _test)

# 预处理、后处理测试，只依赖opencv，mac可以运行
add_test(NAME terror_mixup_inference_test COMMAND terror_mixup_inference_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(terror_mixup_inference_test ${terror_mixup_inference_test_src})
target_link_libraries(terror_mixup_inference_test gtest glog protobuf opencv_core opencv_imgcodecs)

set(all_link_libraries gtest glog protobuf boost_system /opt/caffe/build/install/lib/libcaffe.so opencv_core opencv_imgcodecs tron_common tron_proto zmq)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_library(terror_mixup_forward_test OBJECT ${terror_mixup_forward_test_src})
    target_link_libraries(terror_mixup_forward_test ${all_link_libraries})
else ()
    include_directories("/opt/caffe/build/install/include")
    # tsv测试
    add_executable(terror_mixup_forward_test ${terror_mixup_forward_test_src})
    target_link_libraries(terror_mixup_forward_test ${all_link_libraries})
    # 最终输出的.so
    add_library(inference_terror_mixup ${inference_terror_mixup_src})
    target_link_libraries(inference_terror_mixup ${all_link_libraries})
endif ()
