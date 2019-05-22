file(GLOB tmp *.cpp *.hpp *.cu)
set(${PROJECT_NAME}_lib_common ${${PROJECT_NAME}_lib_common} ${tmp})
set(${PROJECT_NAME}_lib_common ${${PROJECT_NAME}_lib_common} PARENT_SCOPE)

	
