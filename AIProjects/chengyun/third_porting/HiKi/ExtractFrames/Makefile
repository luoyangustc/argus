TARGET=ExtractFrames
OBJS=main.o
CFLAGS=-D__linux__
LDFLAGS=-L . -lPlayCtrl -lAudioRender -lSuperRender -ldl
$(TARGET):$(OBJS)
	g++ -o $(TARGET) $(OBJS) $(LDFLAGS)
main.o:main.cpp
	g++ $(CFLAGS) -c main.cpp

clean:
	rm ExtractFrames main.o
