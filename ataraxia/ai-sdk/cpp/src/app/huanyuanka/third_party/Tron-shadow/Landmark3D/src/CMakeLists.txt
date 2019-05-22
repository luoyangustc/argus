file(GLOB tmp *.cpp *.hpp)
set(${PROJECT_NAME}_lib_src ${${PROJECT_NAME}_lib_src} ${tmp})

set(${PROJECT_NAME}_lib_src ${${PROJECT_NAME}_lib_src} PARENT_SCOPE)

