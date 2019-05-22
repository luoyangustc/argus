FFMPEG_INC_PATH=../../ffmpeg_static/include
FFMPEG_LIB_PATH=../../ffmpeg_static/lib
SRC_PATH=../src/
ONVIF_SRC_PATH=../src/onvif/
CJSON_SRC_PATH=../src/cjson/
LOG_SRC_PATH=../src/log/
LOG_INC_PATH=../src/log
CJSON_INC_PATH=../src/cjson
BASE_INC=../../base_lib/inc
DEVICE_SDK_INC=../../qiniu_dev_net_lib/inc
BASE_LIB=../../base_lib/libs
DEVICE_SDK_LIB=../../qiniu_dev_net_lib/libs

TARGET=SmartBox
OBJS=main.o log4z.o CameraSnapShot.o CBitReader.o GetDeviceInfoFromBZSetting.o http_client.o LoadCfg.o MediaConv.o Onvif_implement.o \
	qiniu_dev_net_porting.o SmartBox_porting.o UAVParser.o dom.o duration.o gsoap_common_porting.o mecevp.o smdevp.o soapC.o \
	soapClient.o stdsoap2.o threads.o wsaapi.o wsseapi.o cJSON.o
INCLUDES=-I $(FFMPEG_INC_PATH) -I $(ONVIF_SRC_PATH) -I $(BASE_INC) -I $(DEVICE_SDK_INC) -I $(CJSON_INC_PATH) -I$(LOG_INC_PATH)
CXXFLAGS=-D__STDC_CONSTANT_MACROS -DWITH_OPENSSL -D_LINUX_ -DOS_LINUX
LDFLAGS=-L $(FFMPEG_LIB_PATH) -L $(BASE_LIB) -L $(DEVICE_SDK_LIB) -lc5_dev_net -lc5_base -lcurl -lavformat -lavcodec -lavutil -lrt -lswscale -lswresample -lpthread -lm -lz -ldl -lcrypto -lssl

$(TARGET):$(OBJS)
	g++ $(CXXCFLAGS) -o $(TARGET) $(OBJS) $(LIBS) $(LDFLAGS)
main.o:$(SRC_PATH)main.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)main.cpp
log4z.o:$(LOG_SRC_PATH)log4z.cpp
	g++ $(CXXCFLAGS) $(INCLUDES) -c $(LOG_SRC_PATH)log4z.cpp
CameraSnapShot.o:$(SRC_PATH)CameraSnapShot.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)CameraSnapShot.cpp
CBitReader.o:$(SRC_PATH)CBitReader.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)CBitReader.cpp
GetDeviceInfoFromBZSetting.o:$(SRC_PATH)GetDeviceInfoFromBZSetting.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)GetDeviceInfoFromBZSetting.cpp
http_client.o:$(SRC_PATH)http_client.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)http_client.cpp
LoadCfg.o:$(SRC_PATH)LoadCfg.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)LoadCfg.cpp
MediaConv.o:$(SRC_PATH)MediaConv.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)MediaConv.cpp
Onvif_implement.o:$(SRC_PATH)Onvif_implement.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)Onvif_implement.cpp
qiniu_dev_net_porting.o:$(SRC_PATH)qiniu_dev_net_porting.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)qiniu_dev_net_porting.cpp
SmartBox_porting.o:$(SRC_PATH)SmartBox_porting.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)SmartBox_porting.cpp
UAVParser.o:$(SRC_PATH)UAVParser.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -c $(SRC_PATH)UAVParser.cpp
dom.o:$(ONVIF_SRC_PATH)dom.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)dom.c
duration.o:$(ONVIF_SRC_PATH)duration.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)duration.c
gsoap_common_porting.o:$(ONVIF_SRC_PATH)gsoap_common_porting.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)gsoap_common_porting.c	
mecevp.o:$(ONVIF_SRC_PATH)mecevp.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)mecevp.c
smdevp.o:$(ONVIF_SRC_PATH)smdevp.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)smdevp.c
soapC.o:$(ONVIF_SRC_PATH)soapC.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)soapC.c	
soapClient.o:$(ONVIF_SRC_PATH)soapClient.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)soapClient.c
stdsoap2.o:$(ONVIF_SRC_PATH)stdsoap2.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)stdsoap2.c
threads.o:$(ONVIF_SRC_PATH)threads.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)threads.c
wsaapi.o:$(ONVIF_SRC_PATH)wsaapi.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)wsaapi.c
wsseapi.o:$(ONVIF_SRC_PATH)wsseapi.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(ONVIF_SRC_PATH)wsseapi.c
cJSON.o:$(CJSON_SRC_PATH)cJSON.c
	gcc $(CXXFLAGS) $(INCLUDES) -c $(CJSON_SRC_PATH)cJSON.c	
clean:
	rm $(TARGET) ${OBJS} -rf
