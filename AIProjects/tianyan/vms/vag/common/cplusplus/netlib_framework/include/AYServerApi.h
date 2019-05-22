
#ifndef __AY_NET_API_H__
#define __AY_NET_API_H__

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include "IServerLogical.h"
enum AYNetServType
{
    AYNET_SERV_TYPE_UDP = 1,
    AYNET_SERV_TYPE_TCP = 2,
    AYNET_SERV_TYPE_HTTP = 3,
};

extern "C" int AYServer_Init(IServerLogical* pServerSink, const char* log_path);
extern "C" int AYServer_OpenServ(int serv_type, const char* serv_ip, unsigned short serv_port);
extern "C" int AYServer_CloseServ(int serv_type, const char* serv_ip, unsigned short serv_port);
extern "C" void AYServer_Run_loop();
extern "C" void AYServer_Clear();
extern "C" std::ostringstream& AYServer_DumpInfo(std::ostringstream& oss);

#endif	//__AY_NET_API_H__
