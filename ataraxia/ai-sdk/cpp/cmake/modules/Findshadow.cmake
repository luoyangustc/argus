include(FindPackageHandleStandardArgs)

set(shadow_ROOT_DIR
    ${AISDK_SOURCE_DIR}/third_party/shadow/build
    CACHE PATH "Folder contains shadow")

set(shadow_DIR ${shadow_ROOT_DIR})

set(shadow_PLATFORM)
set(shadow_ARC)
set(shadow_LIBS)
if (MSVC)
  set(shadow_PLATFORM windows)
  set(shadow_ARC x86_64)
elseif (ANDROID)
  set(shadow_PLATFORM android)
  set(shadow_ARC ${ANDROID_ABI})
elseif (APPLE)
  set(shadow_PLATFORM darwin)
  set(shadow_ARC x86_64)
elseif (UNIX AND NOT APPLE)
  set(shadow_PLATFORM linux)
  set(shadow_ARC x86_64)
endif ()

find_path(shadow_INCLUDE_DIRS
          NAMES core/network.hpp
          PATHS ${shadow_DIR}
          PATH_SUFFIXES include include/x86_64 include/x64
          DOC "shadow include"
          NO_DEFAULT_PATH)

if (NOT MSVC)
  find_library(shadow_LIBRARIES
               NAMES shadow
               PATHS ${shadow_DIR}
               PATH_SUFFIXES lib lib64 lib/${shadow_PLATFORM}/${shadow_ARC}
               DOC "shadow library"
               NO_DEFAULT_PATH)
endif ()

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
