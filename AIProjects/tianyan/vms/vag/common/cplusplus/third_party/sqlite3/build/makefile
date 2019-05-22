ROOT_PATH=../
INCLUDE_PATH=${ROOT_PATH}include/
SRC_PATH=${ROOT_PATH}src/

BASE_PATH=${ROOT_PATH}../../base/

OBJSPATH=./objs/
LIBSPATH=${ROOT_PATH}libs/
LIB_TARGET=${LIBSPATH}libc5_sqlite.a

BASECLASSOBJS=${OBJSPATH}sqlite3.o ${OBJSPATH}SqliteDB.o

CXX=g++
GCC=gcc
CXXFLAGS=-g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG
INCLUDES=-I /usr/local/boost_1_56_0 -I ${INCLUDE_PATH} -I ${BASE_PATH}include

LIBS=-lpthread -ldl -lrt

.PHONY:all
${LIB_TARGET}:${BASECLASSOBJS}
	ar -rv ${LIB_TARGET} ${BASECLASSOBJS}

${OBJSPATH}sqlite3.o:${SRC_PATH}sqlite3.c ${INCLUDE_PATH}sqlite3.h
		$(GCC) ${CXXFLAGS} ${LIBS} -c $< -o $@

${OBJSPATH}SqliteDB.o:${SRC_PATH}SqliteDB.cpp ${INCLUDE_PATH}SqliteDB.h
		$(CXX) ${CXXFLAGS} ${INCLUDES} ${LIBS} -c $< -o $@

.PHONY:clean
clean:
	-rm ${BASECLASSOBJS}  ${LIB_TARGET}  -rf

