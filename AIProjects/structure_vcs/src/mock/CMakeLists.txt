file(GLOB_RECURSE tmp *.cpp *.hpp)
set(structure_vcs_src ${structure_vcs_src} ${tmp})
set(structure_vcs_src ${structure_vcs_src} PARENT_SCOPE)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/include)
add_library(vss SHARED vss.cpp)
target_link_libraries(vss proto)