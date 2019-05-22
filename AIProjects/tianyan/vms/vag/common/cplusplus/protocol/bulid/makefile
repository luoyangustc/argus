ROOT_PATH=../../

BASE_PATH=${ROOT_PATH}base/
PROTOCOL_PATH=${ROOT_PATH}protocol/
BOOST_PATH=${ROOT_PATH}third_party/boost/

INCLUDE_PATH=${PROTOCOL_PATH}include/
SRC_PATH=${PROTOCOL_PATH}src/

OBJSPATH=./objs/
LIBSPATH=${PROTOCOL_PATH}libs/

LIB_TARGET=${LIBSPATH}libc5_protocol.a

CLASSOBJS=\
    ${OBJSPATH}protocol_header.o \
    ${OBJSPATH}protocol_exchangekey.o \
    ${OBJSPATH}protocol_client.o \
    ${OBJSPATH}protocol_device.o \
	${OBJSPATH}protocol_stream.o \
    ${OBJSPATH}protocol_status.o 
    
CXX=g++
CXXFLAGS=-g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG
INCLUDES=-I ${BOOST_PATH}include -I ${INCLUDE_PATH} -I ${BASE_PATH}include

#LIBS=-lpthread -ldl -lrt -lrt ${BASE_PATH}libs/libc5_base.a
#all: ${CLASSOBJS}
.PHONY:all
${LIB_TARGET}:${CLASSOBJS}
	ar -rv ${LIB_TARGET} ${CLASSOBJS}

${OBJSPATH}protocol_header.o:${SRC_PATH}protocol_header.cpp ${INCLUDE_PATH}protocol_header.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}protocol_exchangekey.o:${SRC_PATH}protocol_exchangekey.cpp ${INCLUDE_PATH}protocol_exchangekey.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}protocol_client.o:${SRC_PATH}protocol_client.cpp ${INCLUDE_PATH}protocol_client.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}protocol_device.o:${SRC_PATH}protocol_device.cpp ${INCLUDE_PATH}protocol_device.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}protocol_stream.o:${SRC_PATH}protocol_stream.cpp ${INCLUDE_PATH}protocol_stream.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
    
${OBJSPATH}protocol_status.o:${SRC_PATH}protocol_status.cpp ${INCLUDE_PATH}protocol_status.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY:clean
clean:
	-rm ${CLASSOBJS}  ${LIB_TARGET}  -rf


