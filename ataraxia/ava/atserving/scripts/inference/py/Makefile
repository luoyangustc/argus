CC=g++
CFLAGS :=  -std=c++11 -fPIC
LD_FLAGS := -lstdc++ -lpython2.7 -lprotobuf

LIB = inference.so
INCLUDE_PATH = -I/usr/include/python2.7

all : clean $(LIB)

proto:
	cp ../inference.proto .
	protoc --python_out=. *.proto
	rm inference.proto

$(LIB):
	rm -f $@
	g++ -shared -o $@ *.cc $(INCLUDE_PATH) $(LD_FLAGS) $(CFLAGS)

clean:
	rm -f *.pyc
	rm -f $(LIB)
