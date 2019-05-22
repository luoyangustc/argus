file(GLOB tmp *.cpp *.hpp)
#set(tron_common_src ${tmp})
#set(tron_common_src ${tron_common_src} PARENT_SCOPE)

add_library(tron_common STATIC ${tmp})
target_link_libraries(tron_common
                      PUBLIC glog)
target_include_directories(tron_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
