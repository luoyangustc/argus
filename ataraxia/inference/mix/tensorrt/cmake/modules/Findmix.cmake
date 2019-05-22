include(FindPackageHandleStandardArgs)

set(mix_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/mix CACHE PATH "Folder contains mix")

set(mix_DIR ${mix_ROOT_DIR})

find_path(mix_INCLUDE_DIRS
          NAMES Net.hpp
          PATHS ${mix_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "mix include"
          NO_DEFAULT_PATH)

find_library(mix_LIBRARIES
             NAMES mix
             PATHS ${mix_DIR}
             PATH_SUFFIXES lib lib64 lib/${mix_PLATFORM}/${mix_ARC}
             DOC "mix library"
             NO_DEFAULT_PATH)

find_package_handle_standard_args(mix DEFAULT_MSG mix_INCLUDE_DIRS mix_LIBRARIES)

if (mix_FOUND)
  if (NOT mix_FIND_QUIETLY)
    message(STATUS "Found mix: ${mix_INCLUDE_DIRS}, ${mix_LIBRARIES}")
  endif ()
  mark_as_advanced(mix_ROOT_DIR mix_INCLUDE_DIRS mix_LIBRARIES)
else ()
  if (mix_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find mix")
  endif ()
endif ()
