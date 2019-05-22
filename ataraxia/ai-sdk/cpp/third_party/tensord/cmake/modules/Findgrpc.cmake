
function(GRPC_GENERATE_CPP SRCS HDRS DEST)
  if(NOT ARGN)
    message(SEND_ERROR "Error: GRPC_GENERATE_CPP() called without any proto files")
    return()
  endif()

  if(GRPC_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(DIR ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${DEST}/${FIL_WE}.grpc.pb.cc")
    list(APPEND ${HDRS} "${DEST}/${FIL_WE}.grpc.pb.h")

    add_custom_command(
      OUTPUT "${DEST}/${FIL_WE}.grpc.pb.cc"
             "${DEST}/${FIL_WE}.grpc.pb.h"
      COMMAND protobuf::protoc
      ARGS --grpc_out ${DEST} ${_protobuf_include_path} --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} ${ABS_FIL}
      DEPENDS ${ABS_FIL} protobuf::protoc gRPC::grpc_cpp_plugin
      COMMENT "Running C++ gRPC compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

# By default have GRPC_GENERATE_CPP macro pass -I to protoc
# for each directory where a proto file is referenced.
if(NOT DEFINED GRPC_GENERATE_CPP_APPEND_PATH)
  set(GRPC_GENERATE_CPP_APPEND_PATH TRUE)
endif()

# Find gRPC include directory
find_path(grpc_INCLUDE_DIR grpc/grpc.h)
mark_as_advanced(grpc_INCLUDE_DIR)

# Find gRPC library
find_library(grpc_LIBRARY NAMES grpc)
mark_as_advanced(grpc_LIBRARY)
add_library(gRPC::grpc UNKNOWN IMPORTED)
set_target_properties(gRPC::grpc PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${grpc_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "-lpthread;-ldl"
    IMPORTED_LOCATION ${grpc_LIBRARY}
)

# Find gRPC C++ library
find_library(grpc_grpc++_LIBRARY NAMES grpc++)
mark_as_advanced(grpc_grpc++_LIBRARY)
add_library(gRPC::grpc++ UNKNOWN IMPORTED)
set_target_properties(gRPC::grpc++ PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${grpc_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES gRPC::grpc
    IMPORTED_LOCATION ${grpc_grpc++_LIBRARY}
)

# Find gRPC C++ reflection library
find_library(grpc_grpc++_REFLECTION_LIBRARY NAMES grpc++_reflection)
mark_as_advanced(grpc_grpc++_REFLECTION_LIBRARY)
add_library(gRPC::grpc++_reflection UNKNOWN IMPORTED)
set_target_properties(gRPC::grpc++_reflection PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${grpc_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES gRPC::grpc++
    IMPORTED_LOCATION ${grpc_grpc++_REFLECTION_LIBRARY}
)

# Find gRPC CPP generator
find_program(grpc_CPP_PLUGIN NAMES grpc_cpp_plugin)
mark_as_advanced(grpc_CPP_PLUGIN)
add_executable(gRPC::grpc_cpp_plugin IMPORTED)
set_target_properties(gRPC::grpc_cpp_plugin PROPERTIES
    IMPORTED_LOCATION ${grpc_CPP_PLUGIN}
)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(gRPC DEFAULT_MSG
    grpc_LIBRARY grpc_INCLUDE_DIR grpc_grpc++_REFLECTION_LIBRARY grpc_CPP_PLUGIN)