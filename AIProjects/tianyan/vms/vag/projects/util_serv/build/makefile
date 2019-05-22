ROOT_PATH=../../../
OBJSPATH=objs/
PROTOCOL_PATH=${ROOT_PATH}common/cplusplus
BASE_PATH=${ROOT_PATH}common/cplusplus/base/
NETLIBFW_PATH=${ROOT_PATH}common/cplusplus/netlib_framework/
UTILMODULE_PATH=${ROOT_PATH}common/cplusplus/util_module/
THIRDPARTY_PATH=${ROOT_PATH}common/cplusplus/third_party/

PJR_PATH=../
SERV_TARGET=./util_server

CLASSOBJS=${OBJSPATH}main.o ${OBJSPATH}ServerLogical.o ${OBJSPATH}ServerLogical_http_request.o ${OBJSPATH}variant.o

TCMALLOC_INCLUDE=
TCMALLOC_LIBS=
USE_TCMALLOC=1
ifeq ($(USE_TCMALLOC), 1)
	TCMALLOC_INCLUDE=-I ${THIRDPARTY_PATH}tc_malloc/include/
	TCMALLOC_LIBS=${THIRDPARTY_PATH}/tc_malloc/libs/libtcmalloc.a  \
		${THIRDPARTY_PATH}/tc_malloc/libs/libunwind.a  \
		${THIRDPARTY_PATH}/tc_malloc/libs/libprofiler.a
endif

CXX=g++
GCC=gcc
CXXFLAGS=-Werror -g -O0 -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D __LINUX__ -pthread -D__STDC_CONSTANT_MACROS
ifeq ($(DEBUG), 1)
	CXXFLAGS+=-D__DETAIL_LOG__
endif

INCLUDES=-I ${BASE_PATH}include  \
		-I ${NETLIBFW_PATH}include \
		-I ${UTILMODULE_PATH}include \
		-I ${THIRDPARTY_PATH} 
		${TCMALLOC_INCLUDE}
#need to link libutil_module.so
LIBS=-pthread -ldl -lrt -lrt -lm -lssl\
	${NETLIBFW_PATH}libs/libaynet_server.a  \
	${PROTOCOL_PATH}/protocol/libs/libc5_protocol.a \
	${THIRDPARTY_PATH}http_parser/libs/libhttpparser.a \
	${BASE_PATH}libs/libc5_base.a  \
	${THIRDPARTY_PATH}/json/libs/libjson.a \
	${TCMALLOC_LIBS}  \
	${UTILMODULE_PATH}libs/libutil_module.a
BOOSTLIBS=${THIRDPARTY_PATH}boost/libboost_thread.a \
	${THIRDPARTY_PATH}/boost/libs/libboost_regex.a \
	${THIRDPARTY_PATH}boost/libboost_system.a

.PHONY:all

all: ${SERV_TARGET}

${SERV_TARGET}:${CLASSOBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS} ${BOOSTLIBS} 

${OBJSPATH}main.o:${PJR_PATH}main.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerLogical.o:${PJR_PATH}ServerLogical.cpp ${PJR_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerLogical_http_request.o:${PJR_PATH}ServerLogical_http_request.cpp ${PJR_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}variant.o:${BASE_PATH}src/variant.cpp ${BASE_PATH}include/variant.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY: clean
clean: 
	-rm ${CLASSOBJS} ${SERV_TARGET} -rf
