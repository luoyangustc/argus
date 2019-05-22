TARGET=ExtractVideo
OBJS=main.o
CPPFLAGS=-D__linux__
LDFLAGS=-lPlayCtrl -lAudioRender -lSuperRender -lx264 -ldl
$(TARGET):$(OBJS)
	g++ -o $(TARGET) $(OBJS) $(LDFLAGS)
main.o:main.cpp
	g++ $(CPPFLAGS) -c main.cpp

clean:
	rm ExtractVideo main.o