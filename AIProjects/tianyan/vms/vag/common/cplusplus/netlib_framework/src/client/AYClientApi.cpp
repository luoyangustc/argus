#include "AYClientCore.h"
#include "AYClientApi.h"
#include "Log.h"

CAYClientCore_ptr gpAYClientCore;

int AYClient_Init(const char* log_path)
{
    aynet_log_prepare(EN_LOG_LEVEL_DEBUG, 200*1024*1024, log_path);

    if ( gpAYClientCore )
    {
        WARN_LOG("AYClientCore instance already exists!");
        return 0;
    }

    int ret = 0;
    do
    {
        gpAYClientCore.reset(new CAYClientCore());
        if ( !gpAYClientCore )
        {
            ERROR_LOG("Create AYClientCore instance failed!");
            ret = -1;
            break;
        }

        if ( gpAYClientCore->Init(log_path) < 0 )
        {
            ret = -2;
            break;
        }

        if ( gpAYClientCore->Start() < 0)
        {
            ret = -3;
            break;
        }

        return 0;

    } while (0);

    if (gpAYClientCore)
    {
        gpAYClientCore->Stop();
    }
    return ret;
}

ITCPClient* AYClient_CreateAYTCPClient(void)
{
    ITCPClient* pClient = NULL;
    do 
    {
        if(!gpAYClientCore && (AYClient_Init("./")<0) )
        {
            break;
        }

        pClient = gpAYClientCore->CreateAYTCPClient();

    } while (0);

    return pClient;
}

void AYClient_DestroyAYTCPClient(ITCPClient* pClient)
{
    if(!pClient)
    {
        return;
    }

    pClient->UnadviseSink();
    pClient->Close();

    if(!gpAYClientCore)
    {
        return;
    }

    gpAYClientCore->DestroyAYTCPClient(pClient);
}

void AYClient_Clear()
{
    if(!gpAYClientCore)
    {
        return;
    }
    gpAYClientCore->Stop();
    gpAYClientCore.reset();
}

std::ostringstream& AYClient_DumpInfo(std::ostringstream& oss)
{
    if(!gpAYClientCore)
    {
        return oss;
    }

    return gpAYClientCore->DumpInfo(oss);
}

