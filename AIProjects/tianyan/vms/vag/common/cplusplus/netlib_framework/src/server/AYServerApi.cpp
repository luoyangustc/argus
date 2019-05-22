
#include "AYServerCore.h"
#include "Log.h"
#include "ServerBase.h"
#include "AYServerApi.h"

CAYServerCore_ptr gpAYServerCore;

int AYServer_Init(IServerLogical* pServerSink, const char* log_path)
{
    aynet_log_prepare(EN_LOG_LEVEL_DEBUG, 200*1024*1024, log_path);

    if ( gpAYServerCore )
    {
        WARN_LOG("AYServerCore instance already exists!");
        return 0;
    }

    int ret = 0;
    do
    {
        if (!pServerSink)
        {
            ret = -1;
            break;
        }

        gpAYServerCore.reset(new CAYServerCore());
        if ( !gpAYServerCore )
        {
            ret = -2;
            break;
        }

        if ( gpAYServerCore->Init(pServerSink, log_path ) < 0 )
        {
            ret = -3;
            break;
        }

        if ( gpAYServerCore->Start() < 0)
        {
            ret = -4;
            break;
        }

        return 0;

    } while (0);

    if (gpAYServerCore)
    {
        gpAYServerCore.reset();
    }  

    return ret;
}

int AYServer_OpenServ(int serv_type, const char* serv_ip, unsigned short serv_port)
{
    int ret = 0;
    do 
    {
        if(!gpAYServerCore)
        {
            ret = -1;
            break;
        }

        IServerBase_ptr pServer;
        if(serv_type == AYNET_SERV_TYPE_UDP)
        {
            pServer = gpAYServerCore->CreateServer(en_serv_type_udp, serv_ip, serv_port);
        }
        else if(serv_type == AYNET_SERV_TYPE_TCP)
        {
            pServer = gpAYServerCore->CreateServer(en_serv_type_tcp, serv_ip, serv_port);
        }
        else if(serv_type == AYNET_SERV_TYPE_HTTP)
        {
            pServer = gpAYServerCore->CreateServer(en_serv_type_http, serv_ip, serv_port);
        }
        else
        {
            ret = -2;
            break;
        }

        if(!pServer)
        {
            ret = -3;
            break;
        }

    } while (0);
    
    return ret;
}

int AYServer_CloseServ(int serv_type, const char* serv_ip, unsigned short serv_port)
{
    if(!gpAYServerCore)
    {
        return -1;
    }

    IServerBase_ptr pServer;
    if(serv_type == AYNET_SERV_TYPE_UDP)
    {
        pServer = gpAYServerCore->GetServer(en_serv_type_udp, serv_ip, serv_port);
    }
    else if(serv_type == AYNET_SERV_TYPE_TCP)
    {
        pServer = gpAYServerCore->GetServer(en_serv_type_tcp, serv_ip, serv_port);
    }
    else if(serv_type == AYNET_SERV_TYPE_HTTP)
    {
        pServer = gpAYServerCore->GetServer(en_serv_type_http, serv_ip, serv_port);
    }
    else
    {

    }

    if(pServer)
    {
        gpAYServerCore->DestroyServer(pServer);
    }

    return 0;

}

void AYServer_Run_loop()
{
    if(!gpAYServerCore)
    {
        return;
    }

    gpAYServerCore->RunLoop();
}

void AYServer_Clear()
{
    if(!gpAYServerCore)
    {
        return;
    }

    gpAYServerCore->Stop();
    gpAYServerCore.reset();
}

std::ostringstream& AYServer_DumpInfo(std::ostringstream& oss)
{
    if(!gpAYServerCore)
    {
        return oss;
    }
    return gpAYServerCore->DumpInfo(oss);
}

