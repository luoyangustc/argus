if (Protobuf_FOUND)
  file(DOWNLOAD "http://oxmz2ax9v.bkt.clouddn.com/inference.proto"
       "${PROJECT_SOURCE_DIR}/tron/proto/inference.proto"
       EXPECTED_MD5 a6166b84a09401fd7c4208092f68fcca)
  file(GLOB proto_files "*.proto")
  GenProtobuf(proto_srcs proto_hdrs ${proto_files})
  add_library(tron_proto STATIC ${proto_srcs} ${proto_hdrs})
  if (MSVC)
    set_target_properties(tron_proto PROPERTIES DEBUG_POSTFIX "d")
  endif ()
  # target_include_directories(tron_proto PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
  # set(tron_PROTO_LIB proto PARENT_SCOPE)
endif ()
