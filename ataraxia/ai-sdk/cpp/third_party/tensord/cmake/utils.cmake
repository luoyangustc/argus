
################################################################################

function(download_third_party target)
  configure_file(${TENSORD_SOURCE_DIR}/cmake/${target}-download.CMakeLists.txt.in
                 ${target}-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                  RESULT_VARIABLE result
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${target}-download )
  if(result)
    message(FATAL_ERROR "CMake step for ${target} failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
                  RESULT_VARIABLE result
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${target}-download )
  if(result)
    message(FATAL_ERROR "Build step for ${target} failed: ${result}")
  endif()
endfunction()
