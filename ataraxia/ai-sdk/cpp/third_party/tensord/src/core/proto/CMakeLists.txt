
project(proto CXX)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
set(PROTO_DIR ${INCLUDE_DIR}/tensord/proto)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/tensord.pb.cc"
           "${INCLUDE_DIR}/tensord.pb.h"
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/tensord.proto ${PROTO_DIR}/tensord.proto
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
            --proto_path=${INCLUDE_DIR}
            --cpp_out=${INCLUDE_DIR}
            ${PROTO_DIR}/tensord.proto
    COMMAND mv ${PROTO_DIR}/tensord.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/tensord.pb.cc
    COMMAND rm ${PROTO_DIR}/tensord.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tensord.proto
    COMMENT "Running C++/Python protocol buffer compiler on tensord.proto" VERBATIM)

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/tensord.pb.cc"
    "${PROTO_DIR}/tensord.pb.h"
    PROPERTIES GENERATED TRUE)

add_library(proto STATIC tensord.pb.cc ${PROTO_DIR}/tensord.pb.h)
