ROOT_PATH=../../../
OBJSPATH=objs/
STATUS_SERV_SOURCE_PATH=${ROOT_PATH}projects/status_serv/

STATUS_TARGET=${STATUS_SERV_SOURCE_PATH}build/status_serv

PROTOCOL_PATH=${ROOT_PATH}common/cplusplus/protocol/
BASE_PATH=${ROOT_PATH}common/cplusplus/base/
NETLIB_PATH=${ROOT_PATH}common/cplusplus/netlib_framework/
THIRDPARTY_PATH=${ROOT_PATH}common/cplusplus/third_party/

STATUS_SERV_OBJS=${OBJSPATH}main.o\
	${OBJSPATH}SessionContext.o \
	${OBJSPATH}Device.o \
	${OBJSPATH}DeviceMgr.o \
	${OBJSPATH}RedisAccess.o \
	${OBJSPATH}StatusServer.o \
	${OBJSPATH}HttpRequestHandler.o \
	${OBJSPATH}Stream.o \
	${OBJSPATH}StreamMgr.o \
	${OBJSPATH}SessionMgr.o 

TCMALLOC_INCLUDE=
TCMALLOC_LIBS=
USE_TCMALLOC=1
ifeq ($(USE_TCMALLOC), 1)
	TCMALLOC_INCLUDE=${THIRDPARTY_PATH}tc_malloc/include
	TCMALLOC_LIBS=${THIRDPARTY_PATH}tc_malloc/libs/libtcmalloc.a ${THIRDPARTY_PATH}tc_malloc/libs/libunwind.a ${THIRDPARTY_PATH}tc_malloc/libs/libprofiler.a
endif

CC=gcc
CXX=g++
CFLAGS=-g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D __LINUX__ -D _STATUS_SERVER_
CXXFLAGS=-g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D __LINUX__ -D _STATUS_SERVER_
INCLUDES= \
	-I ${STATUS_SERV_SOURCE_PATH} \
	-I ${ROOT_PATH}common/cplusplus  \
	-I ${ROOT_PATH}common/cplusplus/base/include  \
	-I ${ROOT_PATH}common/cplusplus/netlib_framework/include  \
	-I ${ROOT_PATH}common/cplusplus/protocol/include  \
	-I ${ROOT_PATH}common/cplusplus/util_module/include  \
	-I ${THIRDPARTY_PATH}boost/include \
	-I ${THIRDPARTY_PATH}redis/include \
	-I ${TCMALLOC_INCLUDE}

LIBS=-pthread -ldl -lrt -lm \
	${TCMALLOC_LIBS} \
	${NETLIB_PATH}libs/libaynet_server.a  \
	${PROTOCOL_PATH}libs/libc5_protocol.a \
	${BASE_PATH}libs/libc5_base.a  \
	${ROOT_PATH}common/cplusplus/util_module/libs/libutil_module.a \
	${THIRDPARTY_PATH}json/libs/libjson.a \
	${THIRDPARTY_PATH}redis/libs/libxredis.a \
	${THIRDPARTY_PATH}redis/libs/libhiredis.a \
	${THIRDPARTY_PATH}http_parser/libs/libhttpparser.a

CURLLIBS = -lcurl -lssl -lcrypto

BOOSTLIBS=\
	${THIRDPARTY_PATH}boost/libs/libboost_system.a  \
	${THIRDPARTY_PATH}boost/libs/libboost_thread.a  \
	${THIRDPARTY_PATH}boost/libs/libboost_regex.a \
	${THIRDPARTY_PATH}boost/libs/libboost_program_options.a

.PHONY:all

${STATUS_TARGET}:${STATUS_SERV_OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS} ${CURLLIBS} ${BOOSTLIBS} 

${OBJSPATH}main.o:${STATUS_SERV_SOURCE_PATH}main.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}StatusServer.o:${STATUS_SERV_SOURCE_PATH}StatusServer.cpp ${STATUS_SERV_SOURCE_PATH}StatusServer.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}HttpRequestHandler.o:${STATUS_SERV_SOURCE_PATH}HttpRequestHandler.cpp ${STATUS_SERV_SOURCE_PATH}HttpRequestHandler.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@	

${OBJSPATH}SessionContext.o:${STATUS_SERV_SOURCE_PATH}SessionContext.cpp ${STATUS_SERV_SOURCE_PATH}SessionContext.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}StreamMgr.o:${STATUS_SERV_SOURCE_PATH}StreamMgr.cpp ${STATUS_SERV_SOURCE_PATH}StreamMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}Device.o:${STATUS_SERV_SOURCE_PATH}Device.cpp ${STATUS_SERV_SOURCE_PATH}Device.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
		
${OBJSPATH}DeviceMgr.o:${STATUS_SERV_SOURCE_PATH}DeviceMgr.cpp ${STATUS_SERV_SOURCE_PATH}DeviceMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}RedisAccess.o:${STATUS_SERV_SOURCE_PATH}RedisAccess.cpp ${STATUS_SERV_SOURCE_PATH}RedisAccess.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}SessionMgr.o:${STATUS_SERV_SOURCE_PATH}SessionMgr.cpp ${STATUS_SERV_SOURCE_PATH}SessionMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}Stream.o:${STATUS_SERV_SOURCE_PATH}Stream.cpp ${STATUS_SERV_SOURCE_PATH}Stream.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY:clean
clean:
	-rm ${STATUS_SERV_OBJS} ${STATUS_TARGET} -rf
