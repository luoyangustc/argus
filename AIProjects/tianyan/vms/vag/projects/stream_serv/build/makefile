ROOT_PATH=../../../
OBJSPATH=./objs/
SERV_TARGET=./stream_serv
#SERV_TARGET=${ROOT_PATH}/../build/stream_serv

SRC_PATH=../
CPLUS_PATH=${ROOT_PATH}common/cplusplus/
BASE_PATH=${ROOT_PATH}common/cplusplus/base/
PROTOCOL_PATH=${ROOT_PATH}common/cplusplus/protocol/
NETLIB_PATH=${ROOT_PATH}common/cplusplus/netlib_framework/
TSENCODE_PATH=${ROOT_PATH}common/cplusplus/ts_encode/
THIRD_PARTY_PATH=${ROOT_PATH}common/cplusplus/third_party/
CURL_PATH=${THIRD_PARTY_PATH}curl/
SQLITE_PATH=${THIRD_PARTY_PATH}sqlite3/
FAAC_PATH=${THIRD_PARTY_PATH}faac/
FFMPEG_PATH=${THIRD_PARTY_PATH}ffmpeg/
HTTP_PARSER_PATH=${THIRD_PARTY_PATH}http_parser/
BOOST_PATH=${THIRD_PARTY_PATH}boost/
OPENSSL_PATH=${THIRD_PARTY_PATH}openssl/

SERV_OBJS=\
	${OBJSPATH}main.o\
	${OBJSPATH}Config.o\
	${OBJSPATH}Frame.o\
	${OBJSPATH}FrameMgr.o\
	${OBJSPATH}MediaSessionBase.o\
	${OBJSPATH}MediaSessionLive.o\
	${OBJSPATH}MediaSessionMgr.o\
	${OBJSPATH}MonitorThread.o\
	${OBJSPATH}PuStream.o\
	${OBJSPATH}ServerLogical.o\
	${OBJSPATH}ServerLogical_http_request.o\
	${OBJSPATH}StreamStatusReport.o\
	${OBJSPATH}CuStream.o\
	${OBJSPATH}CloudStorageAgent.o\
	${OBJSPATH}RtmpLive.o

TCMALLOC_INCLUDE=
TCMALLOC_LIBS=
TCMALLOC_FLAG=
USE_TCMALLOC=1
ifeq ($(USE_TCMALLOC), 1)
	TCMALLOC_INCLUDE=${THIRD_PARTY_PATH}tc_malloc/include
	TCMALLOC_LIBS=${THIRD_PARTY_PATH}tc_malloc/libs/libtcmalloc.a  ${THIRD_PARTY_PATH}tc_malloc/libs/libunwind.a ${THIRD_PARTY_PATH}tc_malloc/libs/libprofiler.a
	TCMALLOC_FLAG=-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
endif

CXX=g++
GCC=gcc
CXXFLAGS=-Werror -g -O0 -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D __LINUX__ -pthread -D__STDC_CONSTANT_MACROS ${TCMALLOC_FLAG}

INCLUDES=-I ${BOOST_PATH}include \
	-I ${TCMALLOC_INCLUDE} \
	-I ${SRC_PATH} \
	-I ${CPLUS_PATH} \
	-I ${BASE_PATH}include \
	-I ${PROTOCOL_PATH}include \
	-I ${NETLIB_PATH}include \
	-I ${CURL_PATH}include \
	-I ${THIRD_PARTY_PATH}librtmp/include \
	-I ${HTTP_PARSER_PATH}include


LIBS=-lpthread -ldl -lrt -lrt -lm \
	${TCMALLOC_LIBS} \
	${NETLIB_PATH}libs/libaynet_server.a \
	${NETLIB_PATH}libs/libaynet_client.a \
	${BASE_PATH}libs/libc5_base.a \
	${PROTOCOL_PATH}libs/libc5_protocol.a \
	${THIRD_PARTY_PATH}librtmp/libs/librtmp.a \
	${THIRD_PARTY_PATH}adns/libs/libadns.a \
	${HTTP_PARSER_PATH}libs/libhttpparser.a 

CURLLIBS = -lcurl -lssl -lcrypto

BOOSTLIBS=\
	${BOOST_PATH}libs/libboost_thread.a \
	${BOOST_PATH}libs/libboost_system.a \
	${BOOST_PATH}libs/libboost_regex.a 

.PHONY:all

all: ${SERV_TARGET}

${SERV_TARGET}:${SERV_OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS} ${CURLLIBS} ${BOOSTLIBS}

${OBJSPATH}main.o:${SRC_PATH}main.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}Config.o:${SRC_PATH}Config.cpp ${SRC_PATH}Config.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}Frame.o:${SRC_PATH}Frame.cpp ${SRC_PATH}Frame.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}FrameMgr.o:${SRC_PATH}FrameMgr.cpp ${SRC_PATH}FrameMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}MediaSessionBase.o:${SRC_PATH}MediaSessionBase.cpp ${SRC_PATH}MediaSessionBase.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}MediaSessionLive.o:${SRC_PATH}MediaSessionLive.cpp ${SRC_PATH}MediaSessionLive.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}MediaSessionMgr.o:${SRC_PATH}MediaSessionMgr.cpp ${SRC_PATH}MediaSessionMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}MonitorThread.o:${SRC_PATH}MonitorThread.cpp ${SRC_PATH}MonitorThread.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}PuStream.o:${SRC_PATH}PuStream.cpp ${SRC_PATH}PuStream.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerLogical.o:${SRC_PATH}ServerLogical.cpp ${SRC_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ServerLogical_http_request.o:${SRC_PATH}ServerLogical_http_request.cpp ${SRC_PATH}ServerLogical.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}StreamStatusReport.o:${SRC_PATH}StreamStatusReport.cpp ${SRC_PATH}StreamStatusReport.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}CuStream.o:${SRC_PATH}CuStream.cpp ${SRC_PATH}CuStream.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}CloudStorageAgent.o:${SRC_PATH}CloudStorageAgent.cpp ${SRC_PATH}CloudStorageAgent.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}RtmpLive.o:${SRC_PATH}RtmpLive.cpp ${SRC_PATH}RtmpLive.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY: clean
clean:
	-rm ${SERV_OBJS} ${SERV_TARGET} -rf
