TARGET=DvrDL
OBJS=main.o
CFLAGS=-I ./incCn
LDFLAGS=-L ./lib -L ./lib/HCNetSDKCom -lHCCore -lhpr -lhcnetsdk -lPlayCtrl -lHCCoreDevCfg -lHCIndustry -ldl
$(TARGET):$(OBJS)
	g++ -o $(TARGET) $(OBJS) $(LDFLAGS)
main.o:main.cpp
	g++ $(CFLAGS) -c main.cpp

clean:
	rm -f DvrDL main.o
