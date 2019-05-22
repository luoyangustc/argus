
####################################################################
# Reads set of version defines from the header file
# Usage:
#   parse_header(<file> <define1> <define2> <define3> ..)
function (parse_header FILENAME)
  set(vars_regex "")
  foreach (name ${ARGN})
    if (vars_regex)
      set(vars_regex "${vars_regex}|${name}")
    else ()
      set(vars_regex "${name}")
    endif ()
  endforeach ()
  set(HEADER_CONTENTS "")
  if (EXISTS ${FILENAME})
    file(STRINGS ${FILENAME} HEADER_CONTENTS REGEX "#define[ \t]+(${vars_regex})[ \t]+[0-9]+")
  endif ()
  foreach (name ${ARGN})
    set(num "")
    if (HEADER_CONTENTS MATCHES ".+[ \t]${name}[ \t]+([0-9]+).*")
      string(REGEX REPLACE ".+[ \t]${name}[ \t]+([0-9]+).*" "\\1" num "${HEADER_CONTENTS}")
    endif ()
    set(${name} ${num} PARENT_SCOPE)
  endforeach ()
endfunction ()

####################################################################
# Find current os platform and architecture
# Usage:
#   find_os_arch(<platform_var> <arch_var>)
function (find_os_arch platform_var arch_var)
  set(${platform_var})
  set(${arch_var})
  if (MSVC)
    set(${platform_var} windows PARENT_SCOPE)
    set(${arch_var} x86_64 PARENT_SCOPE)
  elseif (ANDROID)
    set(${platform_var} android PARENT_SCOPE)
    set(${arch_var} ${ANDROID_ABI} PARENT_SCOPE)
  elseif (APPLE)
    set(${platform_var} darwin PARENT_SCOPE)
    set(${arch_var} x86_64 PARENT_SCOPE)
  elseif (UNIX AND NOT APPLE)
    set(${platform_var} linux PARENT_SCOPE)
    set(${arch_var} x86_64 PARENT_SCOPE)
  endif ()
endfunction ()

####################################################################

macro(fetch_external target _project_dir _process_dir)
    set(_PROJECT_ROOT ${_project_dir})
    # set(_DOWNLOAD_ROOT "${_project_dir}/external/${target}")
    set(_BUILD_ROOT ${_process_dir})
    configure_file(
        ${_project_dir}/cmake/${target}-download.cmake
        ${_process_dir}/CMakeLists.txt
        @ONLY
    )
    unset(_PROJECT_ROOT)
    unset(_BUILD_ROOT)

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY ${_process_dir}
    )

    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY ${_process_dir}
    )

endmacro()
