#ifndef __AY_SERVER_CORE_H_
#define __AY_SERVER_CORE_H_
#include <map>

#include "Common.h"
#include "AYNetCoreCfg.h"
#include "IServerLogical.h"
#include "ServerBase.h"
#include "TCPServer.h"
#include "UDPServer.h"
#include "SignalObject.h"

using namespace  boost::asio;

class CAYServerCore
{
public:
	CAYServerCore();
	~CAYServerCore();
	int Init(IServerLogical* pServerSink, const char* pLogPath);
	int Start();
    int Stop();
    int RunLoop();
    int IdleRunProc();
    int OnTimeout(const boost::system::error_code& ec);
    std::ostringstream& DumpInfo(std::ostringstream& oss);
public:
    IServerBase_ptr GetServer(EN_SERV_TYPE serv_type, string serv_ip, uint16 listen_port);
    IServerBase_ptr CreateServer(EN_SERV_TYPE serv_type, string serv_ip, uint16 listen_port);
    int DestroyServer(IServerBase_ptr pServer);
private:
	void StartTimer();
	void StartIdleTask();
    void StartTest();
    void StopTest();
private:
    void UpdateCfg();
private:
    bool                    m_bRunning;
    IServerLogical*			m_pServiceLogic;
    CAYNetCoreCfg           m_Cfg;
    CSignalObject_ptr       m_spSignalObj;

    io_service_pool         m_IoServicePool;
    io_service_pool         m_TaskServicePool;

    IOService_ptr           m_spTimerService;
    IOServiceWork_ptr       m_spTimerServiceWork;
    Timer_ptr               m_spTimer;

    boost::thread_group     m_Threads;

    IServerBaseList         m_ServerList;
    boost::recursive_mutex  m_ServerListLock;

    time_t                  m_nLastUpdateCfgTime;
};

typedef boost::shared_ptr<CAYServerCore>  CAYServerCore_ptr;

#endif //__AY_SERVER_CORE_H_