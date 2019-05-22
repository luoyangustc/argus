#ifndef __AY_CLIENT_CORE_H_
#define __AY_CLIENT_CORE_H_
#include <map>

#include "Common.h"
#include "IServerLogical.h"
#include "IClientSocket.h"
#include "AYNetCoreCfg.h"
#include "IoServicePool.h"

using namespace  boost::asio;

enum EN_TCP_CLIENT_TYPE
{
    en_tcp_client_common,
    en_tcp_client_ay,
};

class CAYClientCore
{
public:
	CAYClientCore();
	~CAYClientCore();	
	int Init(const char* pLogPath);
	int Start();
    int Stop();
    int OnTimeout(const boost::system::error_code& ec);
    int RunLoop();
    std::ostringstream& DumpInfo(std::ostringstream& oss);
public:
    ITCPClient* CreateTCPClient();
    void DestroyTCPClient(ITCPClient* pClient);

    ITCPClient* CreateAYTCPClient();
    void DestroyAYTCPClient(ITCPClient* pClient);
private:
    void StartTimer();
    void UpdateCfg();
private:
    bool                    m_bRunning;

    CAYNetCoreCfg           m_Cfg;

    io_service_pool         m_IoServicePool;
    io_service_pool         m_TaskServicePool;

    IOService_ptr           m_spTimerService;
    IOServiceWork_ptr       m_spTimerServiceWork;
    Timer_ptr               m_spTimer;

    boost::thread_group     m_Threads;

    ITCPClientList          m_TCPClientList;
    boost::recursive_mutex  m_TCPClientListLock;

    time_t                  m_nLastUpdateCfgTime;
};

typedef boost::shared_ptr<CAYClientCore>  CAYClientCore_ptr;

#endif //__AY_CLIENT_CORE_H_