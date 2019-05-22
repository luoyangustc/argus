#ifdef _WINDOWS
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif

#include "AYServerCore.h"
#include <boost/bind.hpp>
#include "Log.h"
#include "ConfigHelper.h"
#include "HTTPSession.h"
#include "AYSession.h"

CAYServerCore::CAYServerCore()
{
}

CAYServerCore::~CAYServerCore()
{
}

int CAYServerCore::Init(IServerLogical* pServerLogical, const char* pLogPath)
{
    do 
    {
        string sConfigDir = CConfigHelper::get_default_config_dir();
        string sCfgName = sConfigDir + "/" + "netcore.conf" ;
        if( m_Cfg.ReadCfg(sCfgName) < 0 )
        {
        }
        m_nLastUpdateCfgTime = time(NULL);
        aynet_log_level_set(m_Cfg.GetLogLevel());
        aynet_log_size_set(m_Cfg.GetLogSize());

        DEBUG_LOG( "CAYServerCore::Init-->(io_service_num=%u,works=%u),(task_service_num=%u,works=%u),(log_level=%u),(log_max_level=%uMB).", 
            m_Cfg.GetServerIoServiceNum(), 
            m_Cfg.GetServerIoServiceWorks(),
            m_Cfg.GetServerTaskServiceNum(), 
            m_Cfg.GetServerTaskServiceWorks(),
            m_Cfg.GetLogLevel(),
            m_Cfg.GetLogSize());

        if( !pServerLogical )
        {
            WARN_LOG("CAYServerCore::Init-->pServerLogical is nil!" );
            break;
        }
        m_pServiceLogic = pServerLogical;
        
        if( m_IoServicePool.Init(m_Cfg.GetServerIoServiceNum(), m_Cfg.GetServerIoServiceWorks()) < 0 )
        {
            ERROR_LOG( "CAYServerCore::Init-->Init IO Service pool failed!" );
            break;
        }

        if( m_TaskServicePool.Init(m_Cfg.GetServerTaskServiceNum(), m_Cfg.GetServerTaskServiceWorks()) < 0 )
        {
            ERROR_LOG( "CAYServerCore::Init-->Init Task Service pool failed!" );
            break;
        }

        m_spTimerService.reset(new boost::asio::io_service);
        if( !m_spTimerService.get() )
        {
            ERROR_LOG("CAYServerCore::Init-->create io_service failed!" );
            break;
        }
        m_spTimerServiceWork.reset(new io_service::work(*m_spTimerService));
        if( !m_spTimerServiceWork.get() )
        {
            ERROR_LOG("CAYServerCore::Init-->create io_service work failed!" );
            break;
        }

        m_spTimer.reset(new boost::asio::deadline_timer(*m_spTimerService));
        if( !m_spTimer.get() )
        {
            ERROR_LOG("CAYServerCore::Init-->create timer failed!" );
            break;
        }

        ERROR_LOG("CAYServerCore::Init-->Success!" );

        return 0;
    } while (0);
    
    return -1;
}

int CAYServerCore::Start()
{
    StartTest();

    m_bRunning = true;

    m_IoServicePool.Start();
    m_TaskServicePool.Start();

    StartTimer();
    StartIdleTask();

    ERROR_LOG("CAYServerCore::start-->Success!" );

    return 0;
}

int CAYServerCore::RunLoop()
{
    m_spSignalObj.reset(new CSignalObject());
    if(!m_spSignalObj)
    {
        return -1;
    }
    m_spSignalObj->wait();
    return 0;
}

int CAYServerCore::Stop()
{
    ERROR_LOG("CAYServerCore::stop-->Enter!" );

    m_bRunning = false;

    if (!m_pServiceLogic)
    {
        m_pServiceLogic->Stop();
    }

    if( m_spTimerService.get() )
    {
        m_spTimerService->stop();
    }
    m_IoServicePool.Stop();
    m_TaskServicePool.Stop();

    m_Threads.join_all();
    m_IoServicePool.Join();
    m_TaskServicePool.Join();

    boost::lock_guard<boost::recursive_mutex> lock(m_ServerListLock);
    IServerBaseListIter itor = m_ServerList.begin();
    while ( itor != m_ServerList.end() )
    {
        IServerBase_ptr pServer = *itor;
        pServer->Stop();
        m_ServerList.erase(itor++);
    }

    StopTest();
    
    ERROR_LOG("CAYServerCore::stop-->Success!" );
    return 0;
}

int CAYServerCore::IdleRunProc()
{
    DEBUG_LOG("CAYServerCore::IdleRunProc-->Enter thread function!" );

    while ( m_bRunning )
    {
        UpdateCfg();

        if (m_pServiceLogic)
        {
            m_pServiceLogic->DoIdleTask();
        }

        boost::this_thread::sleep(boost::posix_time::seconds(10));
    }

    return 0;
}


IServerBase_ptr CAYServerCore::GetServer(EN_SERV_TYPE serv_type, string serv_ip, uint16 listen_port)
{
    IServerBase_ptr pServer;

    //获取server处理，待增加

    return pServer;
}


IServerBase_ptr CAYServerCore::CreateServer(EN_SERV_TYPE serv_type, string serv_ip, uint16 listen_port)
{
    IServerBase_ptr pServer;

    do 
    {
        pServer = GetServer(serv_type, serv_ip, listen_port);
        if( pServer )
        {
            break;
        }

        switch( serv_type )
        {
        case en_serv_type_udp:
            {
                ip::udp::endpoint serv_ep(ip::address::from_string(serv_ip), listen_port);
                pServer = IServerBase_ptr( new CUDPServer(m_IoServicePool.GetIoService(), serv_ep) );
            }        
            break;
        case en_serv_type_tcp:
            {
                ip::tcp::endpoint serv_ep(ip::address::from_string(serv_ip), listen_port);
                pServer = IServerBase_ptr( new CTCPServer<CAYSession>(m_IoServicePool.GetIoService(), m_TaskServicePool, serv_ep) );
            }
            break;
        case en_serv_type_http:
            {
                ip::tcp::endpoint serv_ep(ip::address::from_string(serv_ip), listen_port);
                pServer = IServerBase_ptr( new CTCPServer<CHTTPSession>(m_IoServicePool.GetIoService(), m_TaskServicePool, serv_ep) );
            }
            break;
        default:
            {
                ERROR_LOG("CAYServerCore::CreateServer-->serv_type(%d) is incorrect!", serv_type );
            }
            break;
        }

        if( !pServer )
        {
            ERROR_LOG("CAYServerCore::CreateServer-->Create Server(%u %s:%d) failed!", serv_type, serv_ip.c_str(), listen_port );
            break;
        }

        if ( pServer->Init(m_pServiceLogic) < 0 )
        {
            pServer.reset();
            ERROR_LOG("CAYServerCore::CreateServer-->Init Server(%u %s:%d) failed!", serv_type, serv_ip.c_str(), listen_port );
            break;
        }

        if ( pServer->Start() < 0 )
        {
            pServer.reset();
            ERROR_LOG("CAYServerCore::CreateServer-->Start Server(%u %s:%d) failed!", serv_type, serv_ip.c_str(), listen_port );
            break;
        }

        {
            boost::lock_guard<boost::recursive_mutex> lock(m_ServerListLock);
            m_ServerList.push_back(pServer);
        }    

        DEBUG_LOG("CAYServerCore::CreateServer-->Add Server(%u %s:%d) success, serv_port=%u!", serv_type, serv_ip.c_str(), listen_port );
    } while (0);

    return pServer;
}

int CAYServerCore::DestroyServer(IServerBase_ptr pServer)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_ServerListLock);
        IServerBaseListIter itor = m_ServerList.begin();
        for(; itor != m_ServerList.end(); ++itor)
        {
            if( *itor == pServer )
            {
                pServer->Stop();
                //pServer->Join(); //后续需要追加server join处理
                m_ServerList.erase(itor);
                break;
            }
        }

    } while (0);
    
    return 0;
}

void CAYServerCore::StartTimer()
{
    m_Threads.create_thread(boost::bind(&boost::asio::io_service::run, m_spTimerService));

    m_spTimer->expires_from_now(boost::posix_time::milliseconds(16));
    m_spTimer->async_wait(boost::bind(&CAYServerCore::OnTimeout,
        this,
        boost::asio::placeholders::error));

    //m_Threads.create_thread(boost::bind(&CAYServerCore::OnTimeout, this));
    DEBUG_LOG("CAYServerCore::StartTimer-->Start 10 ms Timer success." );
}

void CAYServerCore::StartIdleTask()
{
    m_Threads.create_thread(boost::bind(&CAYServerCore::IdleRunProc, this));
    DEBUG_LOG("CAYServerCore::StartTimer-->Start Idle Task success." );
}

int CAYServerCore::OnTimeout(const boost::system::error_code& ec)
{
    //DEBUG_LOG( "CAYServerCore::OnTimeout-->Enter." );
    do 
    {
        if (ec == boost::asio::error::operation_aborted)
        {
            DEBUG_LOG( "CAYServerCore::OnTimeout-->error_code=%s.", ec.message().c_str() );
            break;
        }

        if (m_pServiceLogic)
        {
            m_pServiceLogic->Update();
        }

        boost::lock_guard<boost::recursive_mutex> lock(m_ServerListLock);
        IServerBaseListIter itor = m_ServerList.begin();
        for(; itor!= m_ServerList.end(); ++itor)
        {
            (*itor)->Update();
        }

        while(!m_bRunning)
        {
            printf("update-->in sleep...");
            boost::this_thread::sleep(boost::posix_time::seconds(2));
        }

        m_spTimer->expires_from_now(boost::posix_time::milliseconds(16));
        m_spTimer->async_wait(
                boost::bind(&CAYServerCore::OnTimeout,
                this,
                boost::asio::placeholders::error));

        return 0;

    } while (0);
    
    return -1;
}

void CAYServerCore::UpdateCfg()
{
    string cfg_file = m_Cfg.GetCfgFile();
    if(cfg_file.empty())
    {
        return;
    }

    time_t now = time(NULL);
    if(now<m_nLastUpdateCfgTime)
    {
        m_nLastUpdateCfgTime = now;
    }
    else if( (now - m_nLastUpdateCfgTime) >= 60)
    {
        m_nLastUpdateCfgTime = now;

        CConfigHelper cfg;
        if( cfg.read_config_file(cfg_file) != EN_CFG_SUCCESS)
        {
            return;
        }

        uint32 log_level;
        cfg.get_value(log_level, "debug", "log_level", EN_LOG_LEVEL_WARNING);

        if(log_level != m_Cfg.GetLogLevel())
        {
            m_Cfg.SetLogLevel(log_level);
            aynet_log_level_set(log_level);
        }
    }
}

std::ostringstream& CAYServerCore::DumpInfo(std::ostringstream& oss)
{
    oss << "{";
    boost::lock_guard<boost::recursive_mutex> lock(m_ServerListLock);
    IServerBaseListIter itor = m_ServerList.begin();
    {
        oss << (*itor)->GetServerName() <<":";
        (*itor)->DumpInfo(oss);
        ++itor;
    }

    for( ; itor!=m_ServerList.end(); ++itor )
    {
        oss << ",";

        oss << (*itor)->GetServerName() <<":";
        (*itor)->DumpInfo(oss);
        ++itor;
    }
    oss << "}";

    return oss;
}

//boost::shared_ptr<CLogTest> gpLogTest;
void CAYServerCore::StartTest()
{
    //CAYExchangeKeyTest exchange_test;
    //exchange_test.Test();

    /*if(!gpLogTest)
    {
        gpLogTest.reset(new CLogTest());
        if(gpLogTest)
        {
            gpLogTest->Start();
        }
    }*/
}

void CAYServerCore::StopTest()
{
    /*if(gpLogTest)
    {
        gpLogTest->Stop();
    }*/
}