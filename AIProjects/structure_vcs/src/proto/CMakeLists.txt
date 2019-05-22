file(GLOB tmp *.cpp *.hpp)
set(structure_vcs_src ${structure_vcs_src} ${tmp})

set(structure_vcs_src ${structure_vcs_src} PARENT_SCOPE)

if (Protobuf_FOUND)
  file(GLOB proto_files "*.proto")
  GenProtobuf(proto_srcs proto_hdrs ${proto_files})
  add_library(proto SHARED ${proto_srcs} ${proto_hdrs})
  target_link_libraries(proto ${Protobuf_LIBRARIES})
  install(TARGETS proto DESTINATION ${structure_vcs_INSTALL_LIB_PREFIX})
  set(structure_vcs_PROTO_LIB proto PARENT_SCOPE)
endif ()
