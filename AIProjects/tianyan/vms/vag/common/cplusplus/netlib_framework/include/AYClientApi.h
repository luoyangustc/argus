
#ifndef __AY_CLIENT_API_H__
#define __AY_CLIENT_API_H__

#include <string>
#include <sstream>
#include "IClientSocket.h"

extern "C" int AYClient_Init(const char* log_path);
extern "C" ITCPClient* AYClient_CreateAYTCPClient(void);
extern "C" void AYClient_DestroyAYTCPClient(ITCPClient* pClient);
extern "C" void AYClient_Clear();
extern "C" std::ostringstream& AYClient_DumpInfo(std::ostringstream& oss);

#endif	//__AY_CLIENT_API_H__

