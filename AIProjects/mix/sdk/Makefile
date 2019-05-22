CC= g++
CFLAGS= -Wall -O3 -std=c++11

OBJS= censor.o
DYNAMIC_LIBS= libcensor.so
DEMO= demo
INCLUDES= 
LDFLAGS= -Xlinker --unresolved-symbols=ignore-in-shared-libs

all: $(DYNAMIC_LIBS) demo

%.o : %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -fPIC $< -o $@

$(DYNAMIC_LIBS): $(OBJS)
	rm -f $@
	g++   -shared -o $@ $(OBJS) $(LDFLAGS) 
	
demo:
	g++ -g demo.cpp -L. $(LDFLAGS) -lcensor -o demo 

clean:
	rm -f $(OBJS) $(STATIC_LIBS) $(DYNAMIC_LIBS) demo
