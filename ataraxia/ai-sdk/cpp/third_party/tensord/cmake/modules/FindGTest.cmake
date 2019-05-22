
set(SOURCE_DIR ${TENSORD_SOURCE_DIR}/third_party/googletest)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/googletest-build)

find_path(GTEST_INCLUDE_DIR NAMES gtest/gtest.h
          PATHS ${SOURCE_DIR}/googletest/include)
find_path(GMOCK_INCLUDE_DIR NAMES gmock/gmock.h
          PATHS ${SOURCE_DIR}/googlemock/include)
set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR} ${GMOCK_INCLUDE_DIR})
mark_as_advanced(GTEST_INCLUDE_DIR GMOCK_INCLUDE_DIR GTEST_INCLUDE_DIRS)

find_library(GTEST_LIBRARY NAMES gtest gmock
             PATHS ${BUILD_DIR}/lib)
find_library(GTEST_MAIN_LIBRARY NAMES gtest_main gmock_main
             PATHS ${BUILD_DIR}/lib)
set(GTEST_LIBRARIES ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
mark_as_advanced(GTEST_LIBRARY GTEST_MAIN_LIBRARY GTEST_LIBRARIES)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GTest DEFAULT_MSG
    GTEST_INCLUDE_DIR GMOCK_INCLUDE_DIR GTEST_LIBRARY GTEST_MAIN_LIBRARY)