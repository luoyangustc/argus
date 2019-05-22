
#include "TCPClient.h"
#include "AYTCPClient.h"
#include "AYClientCore.h"

#ifdef _WINDOWS
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif

#include <boost/bind.hpp>
#include "Log.h"
#include "ConfigHelper.h"


CAYClientCore::CAYClientCore()
{
}

CAYClientCore::~CAYClientCore()
{
}

int CAYClientCore::Init(const char* pLogPath)
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
        DEBUG_LOG( "(io_service_num=%u,works=%u),(task_service_num=%u,works=%u),(log_level=%u),(log_max_level=%uMB).", 
            m_Cfg.GetClientIoServiceNum(), 
            m_Cfg.GetClientIoServiceWorks(),
            m_Cfg.GetClientTaskServiceNum(), 
            m_Cfg.GetClientTaskServiceWorks(),
            m_Cfg.GetLogLevel(),
            m_Cfg.GetLogSize());
        
        if( m_IoServicePool.Init(m_Cfg.GetClientIoServiceNum(), m_Cfg.GetClientIoServiceWorks()) < 0 )
        {
            ERROR_LOG( "Init IO Service pool failed!" );
            break;
        }

        if( m_TaskServicePool.Init(m_Cfg.GetClientTaskServiceNum(), m_Cfg.GetClientTaskServiceWorks()) < 0 )
        {
            ERROR_LOG( "Init Task Service pool failed!" );
            break;
        }

        m_spTimerService.reset(new boost::asio::io_service);
        if( !m_spTimerService.get() )
        {
            ERROR_LOG("create io_service failed!" );
            break;
        }
        m_spTimerServiceWork.reset(new io_service::work(*m_spTimerService));
        if( !m_spTimerServiceWork.get() )
        {
            ERROR_LOG("create io_service work failed!" );
            break;
        }

        m_spTimer.reset(new boost::asio::deadline_timer(*m_spTimerService));
        if( !m_spTimer.get() )
        {
            ERROR_LOG("create timer failed!" );
            break;
        }

        DEBUG_LOG("Success!" );

        return 0;
    } while (0);
    
    return -1;
}

int CAYClientCore::Start()
{
    m_bRunning = true;

    m_IoServicePool.Start();
    m_TaskServicePool.Start();

    StartTimer();

    ERROR_LOG("CAYNetCore::start-->Success!" );

    return 0;
}

int CAYClientCore::RunLoop()
{
    m_Threads.join_all();
    m_IoServicePool.Join();
    m_TaskServicePool.Join();

    return 0;
}

int CAYClientCore::Stop()
{
    ERROR_LOG("CAYNetCore::stop-->Enter!" );

    m_bRunning = false;

    m_IoServicePool.Stop();
    m_TaskServicePool.Stop();

    if(m_spTimerService)
    {
        m_spTimerService->stop();
    }
    
    m_IoServicePool.Join();
    m_TaskServicePool.Join();

    m_Threads.join_all();

    ERROR_LOG("Success!" );

    return 0;
}

ITCPClient* CAYClientCore::CreateTCPClient()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_TCPClientListLock);
    ITCPClient_ptr pClient = ITCPClient_ptr(new CTCPClient(m_IoServicePool.GetIoService()));
    if(pClient)
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_TCPClientListLock);
        m_TCPClientList.push_back(pClient);
    }
    return pClient.get();
}
void CAYClientCore::DestroyTCPClient(ITCPClient* pClient)
{
    if( !pClient )
    {
        return;
    }

    pClient->Close();

    boost::lock_guard<boost::recursive_mutex> lock(m_TCPClientListLock);
    ITCPClientListIter itor = m_TCPClientList.begin();
    while ( itor != m_TCPClientList.end() )
    {
        if( (*itor).get() == pClient )
        {
            m_TCPClientList.erase(itor);
            break;
        }
        else
        {
            itor++;
        }
    }
}

ITCPClient* CAYClientCore::CreateAYTCPClient()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_TCPClientListLock);
    ITCPClient_ptr pClient = ITCPClient_ptr(new CAYTCPClient(m_IoServicePool.GetIoService()));
    if(pClient)
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_TCPClientListLock);
        m_TCPClientList.push_back(pClient);
    }
    return pClient.get();
}

void CAYClientCore::DestroyAYTCPClient(ITCPClient* pClient)
{
    if( !pClient )
    {
        return;
    }

    pClient->Close();

    boost::lock_guard<boost::recursive_mutex> lock(m_TCPClientListLock);
    ITCPClientListIter itor = m_TCPClientList.begin();
    while ( itor != m_TCPClientList.end() )
    {
        if( (*itor).get() == pClient )
        {
            m_TCPClientList.erase(itor);
            break;
        }
        else
        {
            itor++;
        }
    }
}

void CAYClientCore::StartTimer()
{
    m_Threads.create_thread(boost::bind(&boost::asio::io_service::run, m_spTimerService));

    m_spTimer->expires_from_now(boost::posix_time::milliseconds(1000));
    m_spTimer->async_wait(boost::bind(&CAYClientCore::OnTimeout,
        this,
        boost::asio::placeholders::error));

    DEBUG_LOG("Start 1000 ms Timer success." );
}

int CAYClientCore::OnTimeout(const boost::system::error_code& ec)
{
    //DEBUG_LOG( "CAYNetCore::OnTimeout-->Enter." );
    do 
    {
        if (ec == boost::asio::error::operation_aborted)
        {
            DEBUG_LOG( "CAYNetCore::OnTimeout-->error_code=%s.", ec.message().c_str() );
            break;
        }

        UpdateCfg();

        m_spTimer->expires_from_now(boost::posix_time::milliseconds(16));
        m_spTimer->async_wait(
                boost::bind(&CAYClientCore::OnTimeout,
                this,
                boost::asio::placeholders::error));

        return 0;
    } while (0);
    return -1;
}

void CAYClientCore::UpdateCfg()
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
        cfg.get_value(log_level, "debug", "log_level", EN_LOG_LEVEL_DEBUG);

        if(log_level != m_Cfg.GetLogLevel())
        {
            m_Cfg.SetLogLevel(log_level);
            aynet_log_level_set(log_level);
        }
    }
}

std::ostringstream& CAYClientCore::DumpInfo(std::ostringstream& oss)
{
    oss << "{";
    boost::lock_guard<boost::recursive_mutex> lock(m_TCPClientListLock);
    {
        oss << "\"tcp_client_num\":\"";
        oss << m_TCPClientList.size();
    }
    oss << "}";
    return oss;
}
