#include "AYNetCoreCfg.h"
#include "ConfigHelper.h"
#include "Log.h"

#ifdef _WINDOWS
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

static unsigned cpu_core_count()
{
    unsigned cpu_cnt = 1;
#ifdef _WINDOWS
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    cpu_cnt = si.dwNumberOfProcessors;
#else
    cpu_cnt = get_nprocs();
#endif
    return cpu_cnt?cpu_cnt:1;
}


CAYNetCoreCfg::CAYNetCoreCfg()
{
    m_strCfgFileName = "";

    m_unClientIoServiceNum = 1;
    m_unClientIoServiceWorks = 1;
    m_unClientTaskServiceNum = 1;
    m_unClientTaskServiceWorks = 1;

    m_unServerIoServiceNum = 1;
    m_unServerIoServiceWorks = 2;
    m_unServerTaskServiceNum = cpu_core_count();
    m_unServerTaskServiceWorks = 4;

    m_unLogLevel = EN_LOG_LEVEL_WARNING;
    m_unLogMaxSize = 200;
}

CAYNetCoreCfg::~CAYNetCoreCfg()
{
}

int CAYNetCoreCfg::ReadCfg(const string& cfg_file)
{
    CConfigHelper cfg;
    if( cfg.read_config_file(cfg_file) != EN_CFG_SUCCESS)
    {
        return -1;
    }
    m_strCfgFileName = cfg_file;

    cfg.get_value(m_unClientIoServiceNum, "client_core", "io_service_num", m_unClientIoServiceNum);
    cfg.get_value(m_unClientIoServiceWorks, "client_core", "io_service_works", m_unClientIoServiceWorks);
    cfg.get_value(m_unClientTaskServiceNum, "client_core", "task_service_num", m_unClientTaskServiceNum);
    cfg.get_value(m_unClientTaskServiceWorks, "client_core", "task_service_works", m_unClientTaskServiceWorks);

    cfg.get_value(m_unServerIoServiceNum, "server_core", "io_service_num", m_unServerIoServiceNum);
    cfg.get_value(m_unServerIoServiceWorks, "server_core", "io_service_works", m_unServerIoServiceWorks);
    cfg.get_value(m_unServerTaskServiceNum, "server_core", "task_service_num", m_unServerTaskServiceNum);
    cfg.get_value(m_unServerTaskServiceWorks, "server_core", "task_service_works", m_unServerTaskServiceWorks);

    cfg.get_value(m_unLogLevel, "debug", "log_level", m_unLogLevel);
    cfg.get_value(m_unLogMaxSize, "debug", "log_max_size", m_unLogMaxSize);
    
    return 0;
}
