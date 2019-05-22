ROOT_PATH=../../../../
PRJ_PATH=${ROOT_PATH}projects/netlib_framework/

TEST_SERV_SRC_PATH=${PRJ_PATH}test/echo_serv/

OBJSPATH=./objs/
LIBSPATH=${ROOT_PATH}libs/centos_64/
BIN_TARGET=${PRJ_PATH}bin/echo_serv

TEST_SERV_OBJS=${OBJSPATH}main.o ${OBJSPATH}ServerLogical.o ${OBJSPATH}DaemonUtil.o 

TCMALLOC_INCLUDE=
TCMALLOC_LIBS=
USE_TCMALLOC=1
ifeq ($(USE_TCMALLOC), 1)
	TCMALLOC_INCLUDE=${ROOT_PATH}include/third_party/gperftools-2.4/include/
	TCMALLOC_LIBS=${LIBSPATH}tc_malloc/libtcmalloc.a  ${LIBSPATH}tc_malloc/libunwind.a ${LIBSPATH}tc_malloc/libprofiler.a
endif

BOOST_PATH=-I/usr/local/boost_1_56_0
INCLUDES=${BOOST_PATH} \
		-I ${TCMALLOC_INCLUDE} \
		-I ${ROOT_PATH}include/base \
		-I ${ROOT_PATH}include/third_party \
		-I ${ROOT_PATH}include/netlib_framework
		
LIBS=-pthread -ldl -lrt \
	${LIBSPATH}libzbase64.a  \
	${LIBSPATH}libc3base.a  \
	${LIBSPATH}libnetwork_server.a \
	${LIBSPATH}libhttpparser.a
BOOSTLIBS=${LIBSPATH}libboost_system-gcc44-mt-1_56.a ${LIBSPATH}libboost_thread-gcc44-mt-1_56.a
#BOOSTLIBS=

CXX=g++
CXXFLAGS=-g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG

.PHONY:all
${BIN_TARGET}:${TEST_SERV_OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS} ${BOOSTLIBS} ${TCMALLOC_LIBS}
	
${OBJSPATH}DaemonUtil.o:${TEST_SERV_SRC_PATH}DaemonUtil.cpp ${TEST_SERV_SRC_PATH}DaemonUtil.h 
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}ServerLogical.o:${TEST_SERV_SRC_PATH}ServerLogical.cpp ${TEST_SERV_SRC_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}main.o:${TEST_SERV_SRC_PATH}main.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY:clean
clean:
	-rm ${TEST_SERV_OBJS} ${BIN_TARGET}  -rf


