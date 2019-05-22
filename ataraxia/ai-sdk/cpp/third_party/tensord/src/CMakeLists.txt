
project(tensord_src)

include_directories(".")

add_subdirectory(core)
if (${TENSORD_SERVER})
  add_subdirectory(servers)

  find_package(gflags REQUIRED)

  add_executable(tensord main.cc)
  target_link_libraries(tensord
                        PUBLIC tensord_core
                        PUBLIC server_grpc
                        PUBLIC grpc++
                        PUBLIC glog
                        PUBLIC gflags
                        PUBLIC pthread)
endif ()
