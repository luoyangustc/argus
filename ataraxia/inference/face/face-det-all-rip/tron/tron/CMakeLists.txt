set(tron_src)
set(tron_examples_src)

add_subdirectory(common)
add_subdirectory(examples)
add_subdirectory(proto)
add_subdirectory(serving)

include_directories(".")

add_library(inference ${tron_src})
target_link_libraries(inference ${Tron_LINKER_LIBS})
target_link_libraries(inference ${Tron_PROTO_LIB})
install(TARGETS inference DESTINATION ${Tron_INSTALL_LIB_PREFIX})

foreach (tron_example_src ${tron_examples_src})
  get_filename_component(tron_example_fil_we ${tron_example_src} NAME_WE)
  add_executable(${tron_example_fil_we} ${tron_example_src})
  target_link_libraries(${tron_example_fil_we} inference)
  install(TARGETS ${tron_example_fil_we} DESTINATION ${Tron_INSTALL_BIN_PREFIX})
endforeach ()
