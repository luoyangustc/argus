ROOT_PATH=../../../
OBJSPATH=./objs/
COMMON_PATH=${ROOT_PATH}common/cplusplus/
SERV_SOURCE_PATH=${ROOT_PATH}projects/session_serv/
THIRD_PARTY_PATH=${COMMON_PATH}third_party/

SERV_TARGET=session_serv

SERV_OBJS=${OBJSPATH}main.o \
	${OBJSPATH}MonitorThread.o \
	${OBJSPATH}ServerConfig.o \
	${OBJSPATH}ServerLogical.o \
	${OBJSPATH}ServerLogical_device.o  \
	${OBJSPATH}ServerLogical_client.o \
	${OBJSPATH}ServerLogical_http_request.o\
	${OBJSPATH}StatusReportClient.o\
	${OBJSPATH}DeviceContext.o \
	${OBJSPATH}UserContext.o\
	${OBJSPATH}DeviceMgr.o \
	${OBJSPATH}UserMgr.o \
	${OBJSPATH}MediaSessionMgr.o \
	${OBJSPATH}WebUserContext.o \
	${OBJSPATH}MediaSession.o \
	${OBJSPATH}Snaper.o

TCMALLOC_INCLUDE=
TCMALLOC_LIBS=
USE_TCMALLOC=1
ifeq ($(USE_TCMALLOC), 1)
	TCMALLOC_INCLUDE=-I ${THIRD_PARTY_PATH}tc_malloc/include/gperftools/
	TCMALLOC_LIBS=${THIRD_PARTY_PATH}tc_malloc/libs/libtcmalloc.a  ${THIRD_PARTY_PATH}tc_malloc/libs/libunwind.a ${THIRD_PARTY_PATH}tc_malloc/libs/libprofiler.a
endif

CXX=g++
GCC=gcc
CXXFLAGS=-Werror -g -O0 -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D __LINUX__ -D _AY_NET_LIB_ -D__STDC_CONSTANT_MACROS -pthread -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
ifeq ($(DEBUG), 1)
	CXXFLAGS+=-D__DETAIL_LOG__
endif

INCLUDES=\
	-I ${COMMON_PATH} \
	-I ${COMMON_PATH}base/include \
	-I ${COMMON_PATH}protocol/include \
	-I ${COMMON_PATH}netlib_framework/include \
	-I ${COMMON_PATH}third_party \
	-I ${COMMON_PATH}third_party/boost/include \
	
LIBS=-lpthread -ldl -lrt -lm  \
	${TCMALLOC_LIBS} \
	${COMMON_PATH}netlib_framework/libs/libaynet_server.a \
	${COMMON_PATH}netlib_framework/libs/libaynet_client.a \
	${COMMON_PATH}protocol/libs/libc5_protocol.a \
	${COMMON_PATH}base/libs/libc5_base.a \
	${THIRD_PARTY_PATH}json/libs/libjson.a \
	${THIRD_PARTY_PATH}http_parser/libs/libhttpparser.a
	
CURLLIBS = -lcurl -lssl -lcrypto

BOOSTLIBS=\
	${THIRD_PARTY_PATH}boost/libs/libboost_thread.a \
	${THIRD_PARTY_PATH}boost/libs/libboost_system.a \
	${THIRD_PARTY_PATH}boost/libs/libboost_date_time.a \
	${THIRD_PARTY_PATH}boost/libs/libboost_regex.a
 
.PHONY:all

all: ${SERV_TARGET}

${SERV_TARGET}:${SERV_OBJS} 
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS} ${CURLLIBS} ${BOOSTLIBS}

${OBJSPATH}main.o:${SERV_SOURCE_PATH}main.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
	
${OBJSPATH}MonitorThread.o:${SERV_SOURCE_PATH}MonitorThread.cpp ${SERV_SOURCE_PATH}MonitorThread.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerConfig.o:${SERV_SOURCE_PATH}ServerConfig.cpp ${SERV_SOURCE_PATH}ServerConfig.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerLogical.o:${SERV_SOURCE_PATH}ServerLogical.cpp ${SERV_SOURCE_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerLogical_device.o:${SERV_SOURCE_PATH}ServerLogical_device.cpp ${SERV_SOURCE_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerLogical_http_request.o:${SERV_SOURCE_PATH}ServerLogical_http_request.cpp ${SERV_SOURCE_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
	
${OBJSPATH}ServerLogical_client.o:${SERV_SOURCE_PATH}ServerLogical_client.cpp ${SERV_SOURCE_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}DeviceContext.o:${SERV_SOURCE_PATH}DeviceContext.cpp ${SERV_SOURCE_PATH}DeviceContext.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}StatusReportClient.o:${SERV_SOURCE_PATH}StatusReportClient.cpp ${SERV_SOURCE_PATH}StatusReportClient.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}DeviceMgr.o:${SERV_SOURCE_PATH}DeviceMgr.cpp ${SERV_SOURCE_PATH}DeviceMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

 ${OBJSPATH}UserMgr.o:${SERV_SOURCE_PATH}UserMgr.cpp ${SERV_SOURCE_PATH}UserMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
 
${OBJSPATH}TokenMgr.o:${SERV_FRAME_PATH}TokenMgr.cpp ${SERV_FRAME_PATH}TokenMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}UserContext.o:${SERV_SOURCE_PATH}UserContext.cpp ${SERV_SOURCE_PATH}UserContext.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}MediaSessionMgr.o:${SERV_SOURCE_PATH}MediaSessionMgr.cpp ${SERV_SOURCE_PATH}MediaSessionMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}WebUserContext.o:${SERV_SOURCE_PATH}WebUserContext.cpp ${SERV_SOURCE_PATH}WebUserContext.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}MediaSession.o:${SERV_SOURCE_PATH}MediaSession.cpp ${SERV_SOURCE_PATH}MediaSession.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}Snaper.o:${SERV_SOURCE_PATH}Snaper.cpp ${SERV_SOURCE_PATH}Snaper.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY: clean
clean: 
	-rm ${SERV_OBJS} ${SERV_TARGET} -rf
