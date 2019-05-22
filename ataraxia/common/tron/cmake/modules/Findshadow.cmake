include(FindPackageHandleStandardArgs)

set(shadow_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/shadow/build CACHE PATH "Folder contains shadow")

set(shadow_DIR ${shadow_ROOT_DIR})

find_path(shadow_INCLUDE_DIRS
          NAMES core/network.hpp
          PATHS ${shadow_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "shadow include"
          NO_DEFAULT_PATH)

find_library(shadow_LIBRARIES
             NAMES shadow
             PATHS ${shadow_DIR}
             PATH_SUFFIXES lib lib64
             DOC "shadow library"
             NO_DEFAULT_PATH)

find_package_handle_standard_args(shadow DEFAULT_MSG shadow_INCLUDE_DIRS shadow_LIBRARIES)

if (shadow_FOUND)
  if (NOT shadow_FIND_QUIETLY)
    message(STATUS "Found shadow: ${shadow_INCLUDE_DIRS}, ${shadow_LIBRARIES}")
  endif ()
  mark_as_advanced(shadow_ROOT_DIR shadow_INCLUDE_DIRS shadow_LIBRARIES)
else ()
  if (shadow_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find shadow")
  endif ()
endif ()
