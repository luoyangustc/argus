ROOT_PATH=../

SRC_PATH=${ROOT_PATH}src/
INC_PATH=${ROOT_PATH}include/
BASE_PATH=${ROOT_PATH}../base/
PROTOCOL_PATH=${ROOT_PATH}../protocol/
HTTP_PARSER_PATH=${ROOT_PATH}../third_party/http_parser/
BOOST_PATH=${ROOT_PATH}../third_party/boost/

COMM_SRC_PATH=${SRC_PATH}comm/
EXCHANGE_KEY_PATH=${SRC_PATH}exchang_key/
SERVER_SRC_PATH=${SRC_PATH}server/
CLIENT_SRC_PATH=${SRC_PATH}client/

LIBSPATH=${ROOT_PATH}libs/

TARGET_CLIENT=${LIBSPATH}libaynet_client.a
TARGET_SERVER=${LIBSPATH}libaynet_server.a

OBJSPATH=./objs/

BASE_OBJS=${OBJSPATH}url_helper.o
EXCHANGE_OBJS=${OBJSPATH}AYExchangeKey.o \
			${OBJSPATH}AYCrypt.o
COMM_OBJS=${OBJSPATH}Log.o \
            ${OBJSPATH}AYMsgReader.o \
            ${OBJSPATH}AYNetCoreCfg.o \
			${OBJSPATH}TCPHandler.o \
			${OBJSPATH}HTTPParser.o \
			${OBJSPATH}SignalObject.o
SERVER_OBJS=${OBJSPATH}AYSession.o \
			${OBJSPATH}HTTPSession.o  \
			${OBJSPATH}UDPServer.o \
			${OBJSPATH}HTTPResponse.o \
            ${OBJSPATH}AYServerCore.o \
            ${OBJSPATH}AYServerApi.o
CLIENT_OBJS=${OBJSPATH}TCPClient.o \
			${OBJSPATH}AYTCPClient.o \
            ${OBJSPATH}AYClientCore.o \
			${OBJSPATH}AYClientApi.o
CXX=g++
AR=ar
CXXFLAGS=-g -fPIC -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D _LINUX_
INCLUDES=-I ${BOOST_PATH}include -I ${SRC_PATH} -I ${INC_PATH} -I ${BASE_PATH}include -I ${BASE_PATH}include/encry -I ${PROTOCOL_PATH}include\
	-I ${HTTP_PARSER_PATH}include -I ${EXCHANGE_KEY_PATH} -I ${COMM_SRC_PATH} -I ${CLIENT_SRC_PATH} -I ${SERVER_SRC_PATH}
LIBS=-lpthread -ldl -lrt ${BASE_PATH}libs/libc5_base.a ${PROTOCOL_PATH}libs/libc5_protocol.a ${PROTOCOL_PATH}libs/libc5_protocol.a  ${HTTP_PARSER_PATH}libs/libhttpparser.a
BOOSTLIBS=${BOOST_PATH}libs/libboost_thread-gcc44-mt-1_56.a ${BOOST_PATH}libs/libboost_system-gcc44-mt-1_56.a

#LINKFLAGS=-shared

.PHONY:all
${LIB_TARGET}:${CLASSOBJS}
	${AR} -rv ${LIB_TARGET} ${CLASSOBJS}
all:${TARGET_SERVER} ${TARGET_CLIENT}
server:${TARGET_SERVER}
${TARGET_SERVER}:${BASE_OBJS} ${COMM_OBJS} ${EXCHANGE_OBJS} ${SERVER_OBJS}
	${AR} rcsv $@ $^
client:${TARGET_CLIENT}
${TARGET_CLIENT}:${BASE_OBJS} ${COMM_OBJS} ${EXCHANGE_OBJS} ${CLIENT_OBJS}
	${AR} rcsv $@ $^
    
#base objs
${OBJSPATH}url_helper.o:${BASE_PATH}src/url_helper.cpp ${BASE_PATH}include/url_helper.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

#exchange key objs
${OBJSPATH}AYExchangeKey.o:${EXCHANGE_KEY_PATH}AYExchangeKey.cpp ${EXCHANGE_KEY_PATH}AYExchangeKey.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYCrypt.o:${EXCHANGE_KEY_PATH}AYCrypt.cpp ${EXCHANGE_KEY_PATH}AYCrypt.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

#comm objs
${OBJSPATH}Log.o:${COMM_SRC_PATH}Log.cpp ${COMM_SRC_PATH}Log.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYMsgReader.o:${COMM_SRC_PATH}AYMsgReader.cpp ${COMM_SRC_PATH}AYMsgReader.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYNetCoreCfg.o:${COMM_SRC_PATH}AYNetCoreCfg.cpp ${COMM_SRC_PATH}AYNetCoreCfg.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@   
${OBJSPATH}TCPHandler.o:${COMM_SRC_PATH}TCPHandler.cpp ${COMM_SRC_PATH}TCPHandler.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}HTTPParser.o:${COMM_SRC_PATH}HTTPParser.cpp ${COMM_SRC_PATH}HTTPParser.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}SignalObject.o:${COMM_SRC_PATH}SignalObject.cpp ${COMM_SRC_PATH}SignalObject.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

#server objs
${OBJSPATH}AYSession.o:${SERVER_SRC_PATH}AYSession.cpp ${SERVER_SRC_PATH}AYSession.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}HTTPSession.o:${SERVER_SRC_PATH}HTTPSession.cpp ${SERVER_SRC_PATH}HTTPSession.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}UDPServer.o:${SERVER_SRC_PATH}UDPServer.cpp ${SERVER_SRC_PATH}UDPServer.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}HTTPResponse.o:${SERVER_SRC_PATH}HTTPResponse.cpp ${SERVER_SRC_PATH}HTTPResponse.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYServerCore.o:${SERVER_SRC_PATH}AYServerCore.cpp ${SERVER_SRC_PATH}/AYServerCore.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYServerApi.o:${SERVER_SRC_PATH}AYServerApi.cpp ${INC_PATH}AYServerApi.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

#client objs
${OBJSPATH}TCPClient.o:${CLIENT_SRC_PATH}TCPClient.cpp ${CLIENT_SRC_PATH}TCPClient.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYTCPClient.o:${CLIENT_SRC_PATH}AYTCPClient.cpp ${CLIENT_SRC_PATH}AYTCPClient.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYClientCore.o:${CLIENT_SRC_PATH}AYClientCore.cpp ${CLIENT_SRC_PATH}/AYClientCore.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
${OBJSPATH}AYClientApi.o:${CLIENT_SRC_PATH}AYClientApi.cpp ${INC_PATH}AYClientApi.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY:clean
clean:
	-rm  ${BASE_OBJS} ${COMM_OBJS} ${EXCHANGE_OBJS} ${SERVER_OBJS} ${CLIENT_OBJS} ${TARGET_CLIENT} ${TARGET_SERVER}  -rf
