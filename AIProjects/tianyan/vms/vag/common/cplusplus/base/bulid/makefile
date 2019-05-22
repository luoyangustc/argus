ROOT_PATH=../

SRC_PATH=${ROOT_PATH}src/
INC_PATH=${ROOT_PATH}include/
ENCRY_SRC_PATH=${ROOT_PATH}src/encry/
ENCRY_INC_PATH=${ROOT_PATH}include/encry/
THIRD_PARTY_PATH=${ROOT_PATH}../third_party/

LIBSPATH=${ROOT_PATH}libs/
LIB_TARGET=${LIBSPATH}libc5_base.a
OBJSPATH=./objs/

BASECLASSOBJS= \
	${OBJSPATH}crc8.o \
    ${OBJSPATH}crc16.o \
    ${OBJSPATH}crc32.o \
	${OBJSPATH}DHCryptLib.o \
	${OBJSPATH}DiffieHellman.o \
    ${OBJSPATH}md5.o \
    ${OBJSPATH}sha1.o \
	${OBJSPATH}TokenCrypt.o \
    ${OBJSPATH}common_thread_base.o \
	${OBJSPATH}common_thread_group.o \
	${OBJSPATH}ConfigHelper.o \
	${OBJSPATH}DaemonUtil.o \
	${OBJSPATH}DeviceID.o \
	${OBJSPATH}GetTickCount.o \
	${OBJSPATH}hex_encode.o \
	${OBJSPATH}http_header_util.o \
	${OBJSPATH}httpdownloader.o  \
	${OBJSPATH}json_help.o \
	${OBJSPATH}LBitField.o \
	${OBJSPATH}logging_posix.o \
    ${OBJSPATH}ParamParser.o \
	${OBJSPATH}SimpleTrafficStat.o \
	${OBJSPATH}tick.o \
	${OBJSPATH}TokenMgr.o \
	${OBJSPATH}vdcharsetutil.o \
	${OBJSPATH}web_request.o \
	${OBJSPATH}ZBase64.o \
	${OBJSPATH}LFile.o \
	${OBJSPATH}variant.o \
	${OBJSPATH}url_helper.o 

CXX=g++
CXXFLAGS=-g -fPIC -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG -D _LINUX_
INCLUDES=\
	-I ${THIRD_PARTY_PATH}boost/include \
	-I ${THIRD_PARTY_PATH} \
	-I ${THIRD_PARTY_PATH}openssl/include \
	-I ${INC_PATH} -I ${ENCRY_INC_PATH} \
	-I../../
#LIBS=-lpthread -ldl -lrt ${THIRD_PARTY_PATH}json/libs/libjson.a ${THIRD_PARTY_PATH}curl/libs/libcurl.a
#BOOSTLIBS=${LIBSPATH}libboost_thread.a ${LIBSPATH}libboost_system.a

#all: ${BASECLASSOBJS}
.PHONY:all
${LIB_TARGET}:${BASECLASSOBJS}
	ar -rv ${LIB_TARGET} ${BASECLASSOBJS}

${OBJSPATH}crc8.o:${ENCRY_SRC_PATH}crc8.cpp ${ENCRY_INC_PATH}crc8.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}crc16.o:${ENCRY_SRC_PATH}crc16.cpp ${ENCRY_INC_PATH}crc16.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}crc32.o:${ENCRY_SRC_PATH}crc32.cpp ${ENCRY_INC_PATH}crc32.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}DHCryptLib.o:${ENCRY_SRC_PATH}DHCryptLib.cpp ${ENCRY_INC_PATH}DHCryptLib.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}DiffieHellman.o:${ENCRY_SRC_PATH}DiffieHellman.cpp ${ENCRY_INC_PATH}DiffieHellman.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}md5.o:${ENCRY_SRC_PATH}md5.cpp ${ENCRY_INC_PATH}md5.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}sha1.o:${ENCRY_SRC_PATH}sha1.cpp ${ENCRY_INC_PATH}sha1.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}TokenCrypt.o:${ENCRY_SRC_PATH}TokenCrypt.cpp ${ENCRY_INC_PATH}TokenCrypt.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}common_thread_base.o:${SRC_PATH}common_thread_base.cpp ${INC_PATH}common_thread_base.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@
	

${OBJSPATH}common_thread_group.o:${SRC_PATH}common_thread_group.cpp ${INC_PATH}common_thread_group.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}ConfigHelper.o:${SRC_PATH}ConfigHelper.cpp ${INC_PATH}ConfigHelper.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}DaemonUtil.o:${SRC_PATH}DaemonUtil.cpp ${INC_PATH}DaemonUtil.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}DeviceID.o:${SRC_PATH}DeviceID.cpp ${INC_PATH}DeviceID.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}GetTickCount.o:${SRC_PATH}GetTickCount.cpp ${INC_PATH}GetTickCount.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}hex_encode.o:${SRC_PATH}hex_encode.cpp ${INC_PATH}hex_encode.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}httpdownloader.o:${SRC_PATH}httpdownloader.cpp ${INC_PATH}httpdownloader.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}http_header_util.o:${SRC_PATH}http_header_util.cpp ${INC_PATH}http_header_util.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}json_help.o:${SRC_PATH}json_help.cpp ${INC_PATH}json_help.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}LBitField.o:${SRC_PATH}LBitField.cpp ${INC_PATH}LBitField.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}logging_posix.o:${SRC_PATH}logging_posix.cpp ${INC_PATH}logging_posix.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}ParamParser.o:${SRC_PATH}ParamParser.cpp ${INC_PATH}ParamParser.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}SimpleTrafficStat.o:${SRC_PATH}SimpleTrafficStat.cpp ${INC_PATH}SimpleTrafficStat.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}tick.o:${SRC_PATH}tick.cpp ${INC_PATH}tick.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}TokenMgr.o:${SRC_PATH}TokenMgr.cpp ${INC_PATH}TokenMgr.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}vdcharsetutil.o:${SRC_PATH}vdcharsetutil.cpp ${INC_PATH}vdcharsetutil.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@


${OBJSPATH}web_request.o:${SRC_PATH}web_request.cpp ${INC_PATH}web_request.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}LFile.o:${SRC_PATH}LFile.cpp ${INC_PATH}LFile.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}ZBase64.o:${SRC_PATH}ZBase64.cpp ${INC_PATH}ZBase64.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}variant.o:${SRC_PATH}variant.cpp ${INC_PATH}variant.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}url_helper.o:${SRC_PATH}url_helper.cpp ${INC_PATH}url_helper.h
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

.PHONY:clean
clean:
	-rm ${BASECLASSOBJS}  ${LIB_TARGET}  -rf


