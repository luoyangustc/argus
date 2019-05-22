
project (grpc CXX)

find_package(grpc 1.18.0 REQUIRED)

execute_process(COMMAND  which grpc_cpp_plugin OUTPUT_VARIABLE GRPC_CPP_PLUGIN)
string(STRIP ${GRPC_CPP_PLUGIN} GRPC_CPP_PLUGIN)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/tensord.pb.cc"
           "${CMAKE_CURRENT_SOURCE_DIR}/tensord.pb.h"
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
            --proto_path=${TENSORD_SOURCE_DIR}/src
            --cpp_out=${TENSORD_SOURCE_DIR}/src
            --grpc_out=${TENSORD_SOURCE_DIR}/src
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/tensord.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tensord.proto
    COMMENT "Running gRPC compiler on tensord.proto" VERBATIM)

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/tensord.pb.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/tensord.pb.h"
    PROPERTIES GENERATED TRUE)

add_library(server_grpc STATIC
            tensord.cc tensord.hpp
            tensord.pb.cc tensord.pb.h
            tensord.grpc.pb.cc tensord.grpc.pb.h)
target_link_libraries(server_grpc
            PRIVATE tensord_core
            PRIVATE protobuf
            PRIVATE glog
            PRIVATE proto
            PRIVATE grpc)
install(TARGETS server_grpc DESTINATION lib)