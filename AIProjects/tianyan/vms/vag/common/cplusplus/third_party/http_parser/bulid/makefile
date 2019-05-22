ROOT_PATH=../
INCLUDE_PATH=${ROOT_PATH}include/
SRC_PATH=${ROOT_PATH}src/

OBJSPATH=./objs/
LIBSPATH=${ROOT_PATH}libs/
LIB_TARGET_A=${LIBSPATH}libhttpparser.a
LIB_TARGET_S=${LIBSPATH}libhttpparser.so

BASECLASSOBJS_A=${OBJSPATH}http_parser_a.o
BASECLASSOBJS_S=${OBJSPATH}http_parser_s.o

CXX=g++
CXXFLAGS=-g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D _LINUX_
INCLUDES=-I ${INCLUDE_PATH}

#LIBS=-lpthread -ldl -lrt
LIBS=

all: ${LIB_TARGET_A} ${LIB_TARGET_S}

${LIB_TARGET_A}:${BASECLASSOBJS_A}
	ar -rv ${LIB_TARGET_A} ${BASECLASSOBJS_A}

${LIB_TARGET_S}:${BASECLASSOBJS_S}
	${CXX} ${CXXFLAGS} -shared $^ -o $@ ${LIBS}

${OBJSPATH}http_parser_a.o:${SRC_PATH}http_parser.cpp ${INCLUDE_PATH}http_parser.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}http_parser_s.o:${SRC_PATH}http_parser.cpp ${INCLUDE_PATH}http_parser.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -fPIC -c $< -o $@
    
.PHONY:clean
clean:
	-rm ${LIB_TARGET_A} ${LIB_TARGET_S}  ${LIB_TARGET_A} ${LIB_TARGET_s}  -rf


