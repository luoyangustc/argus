JSONPATH=./
OBJSPATH=./
LIBSPATH=../libs/
CXXFLAGS += -g -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG
MACROS=-D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D NDEBUG
INCLUDES=-I ./
LDFLAGS += -lpthread
CXX=g++
JSONLIB=$(LIBSPATH)libjson.a
JSONOBJ=${OBJSPATH}json_reader.o ${OBJSPATH}json_writer.o ${OBJSPATH}json_value.o

all:${JSONOBJ}
.PHONY:all
	
${OBJSPATH}json_reader.o:${JSONPATH}json_reader.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}json_writer.o:${JSONPATH}json_writer.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${OBJSPATH}json_value.o:${JSONPATH}json_value.cpp
	$(CXX) ${CXXFLAGS} ${INCLUDES} -c $< -o $@

	ar -rv  $(JSONLIB) ${JSONOBJ}

clean:
	rm ${JSONOBJ} $(JSONLIB) -rf
