TARGET=Onvif_Discover
OBJS=main.o stdsoap2.o dom.o duration.o mecevp.o smdevp.o soapC.o soapClient.o threads.o wsaapi.o wsseapi.o device_account_info.o gsoap_common_porting.o httpd_inner.o cJSON.o
CFLAGS=-D__linux_ -DWITH_OPENSSL
LDFLAGS=-lssl -lcrypto -lm -pthread
$(TARGET):$(OBJS)
	gcc $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)
main.o:main.c
	gcc  $(CFLAGS) -c main.c
stdsoap2.o:stdsoap2.c
	gcc  $(CFLAGS) -c stdsoap2.c
dom.o:dom.c
	gcc  $(CFLAGS) -c dom.c
duration.o:duration.c
	gcc  $(CFLAGS) -c duration.c
mecevp.o:mecevp.c
	gcc  $(CFLAGS) -c mecevp.c
smdevp.o:smdevp.c
	gcc  $(CFLAGS) -c smdevp.c
soapC.o:soapC.c
	gcc  $(CFLAGS) -c soapC.c
soapClient.o:soapClient.c
	gcc  $(CFLAGS) -c soapClient.c
threads.o:threads.c
	gcc  $(CFLAGS) -c threads.c
wsaapi.o:wsaapi.c
	gcc  $(CFLAGS) -c wsaapi.c
wsseapi.o:wsseapi.c
	gcc  $(CFLAGS) -c wsseapi.c
device_account_info.o:device_account_info.c
	gcc  $(CFLAGS) -c device_account_info.c
gsoap_common_porting.o:gsoap_common_porting.c
	gcc  $(CFLAGS) -c gsoap_common_porting.c
httpd_inner.o:httpd_inner.c
	gcc $(CFLAGS) -c httpd_inner.c
cJSON.o:cJSON.c
	gcc $(CFLAGS) -c cJSON.c

clean:
	rm $(TARGET) $(OBJS)
