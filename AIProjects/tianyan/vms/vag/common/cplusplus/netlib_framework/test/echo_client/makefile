ROOT_PATH=../../../
TEST_CLI_SRC_PATH=${ROOT_PATH}netlib_framework/test/echo_client/

OBJSPATH=./objs/
LIBSPATH=${ROOT_PATH}libs/centos_64/
BIN_TARGET=${ROOT_PATH}bin/echo_client

TEST_CLI_OBJS=${OBJSPATH}main.o ${OBJSPATH}AYClient.o

CXX=g++
CXXFLAGS=-g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG
INCLUDES=-I ${ROOT_PATH}base/include -I ${ROOT_PATH}netlib_framework/include -I ${ROOT_PATH}third_party

LIBS=-lpthread -ldl -lrt ${LIBSPATH}libbase.a -L ${LIBSPATH} -lnetwork_client
BOOSTLIBS=${LIBSPATH}libboost_system-gcc44-mt-1_56.so.1.56.0 ${LIBSPATH}libboost_thread-gcc44-mt-1_56.so.1.56.0

.PHONY:all
${BIN_TARGET}:${TEST_CLI_OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS} ${BOOSTLIBS}

${OBJSPATH}main.o:${TEST_CLI_SRC_PATH}main.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}AYClient.o:${TEST_CLI_SRC_PATH}AYClient.cpp ${TEST_CLI_SRC_PATH}AYClient.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY:clean
clean:
	-rm ${TEST_CLI_OBJS} ${BIN_TARGET}  -rf


