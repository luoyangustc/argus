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
# Download external process
# Usage:
#   download_external(<download_config_file> <process_dir>)
function (download_external download_config_file process_dir)
  configure_file(${download_config_file} "${process_dir}/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                  WORKING_DIRECTORY "${process_dir}")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
                  WORKING_DIRECTORY "${process_dir}")
endfunction ()
