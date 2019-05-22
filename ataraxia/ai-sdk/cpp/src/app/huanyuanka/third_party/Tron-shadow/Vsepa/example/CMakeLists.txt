file(GLOB tmp *.cpp *.hpp)
set(${PROJECT_NAME}_examples_src ${${PROJECT_NAME}_examples_src} ${tmp})

set(${PROJECT_NAME}_examples_src ${${PROJECT_NAME}_examples_src} PARENT_SCOPE)
