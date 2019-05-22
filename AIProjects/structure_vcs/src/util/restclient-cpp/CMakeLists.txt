file(GLOB_RECURSE tmp *.cc)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/include)
add_library(restclient-cpp STATIC ${tmp})
target_link_libraries(restclient-cpp curl)