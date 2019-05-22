file(GLOB_RECURSE tmp *.cpp *.hpp)
set(structure_vcs_src ${structure_vcs_src} ${tmp})

set(structure_vcs_src ${structure_vcs_src} PARENT_SCOPE)

add_subdirectory(restclient-cpp)

add_library(structure_vcs_util STATIC ${tmp})