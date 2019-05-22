
project(core CXX)

add_subdirectory(proto)
list(APPEND TENSORD_PLATFORMS_LINK)
list(APPEND TENSORD_PLATFORMS_DEFINITION)
add_subdirectory(platforms)

add_library(tensord_core STATIC
  load_model.cc
  net_list.cc
  tasks.hpp
  engine_impl.hpp
  engines.cc)
target_compile_definitions(tensord_core
                           PUBLIC ${TENSORD_PLATFORMS_DEFINITION})
target_link_libraries(tensord_core
                      PRIVATE protobuf
                      PRIVATE glog
                      PRIVATE proto
                      ${TENSORD_PLATFORMS_LINK})
install(TARGETS tensord_core DESTINATION lib)