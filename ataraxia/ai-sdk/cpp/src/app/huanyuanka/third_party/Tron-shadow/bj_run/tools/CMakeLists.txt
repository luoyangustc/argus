file(GLOB tmp *.cpp *.hpp)
set(${PROJECT_NAME}_tools_src ${${PROJECT_NAME}_tools_src} ${tmp})

set(${PROJECT_NAME}_tools_src ${${PROJECT_NAME}_tools_src} PARENT_SCOPE)
