#include "StatusServer.h"
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include "base/include/DaemonUtil.h"
#include "base/include/ConfigHelper.h"
#include "base/include/logging_posix.h"
#include "netlib_framework/include/AYServerApi.h"
#include "netlib_framework/include/IServerLogical.h"

//#include "Version.h"
#define CURRENT_SERVER_NAME "status_serv"
#define CURRENT_SERVER_VERSION "1.0.0.1"

static int  Usage(int argc, char* argv[], bool& is_daemon);
static int  RegisterSomeSignal();
static void SetCoreLimit();
static void SetLoglevel(int log_level);
static int  StartService();
static void StopProg(int signo);


int main(int argc, char *argv[])
{
    setlocale(LC_ALL,"zh_CN.UTF-8");

    bool is_daemon = true;  
    if ( Usage(argc, argv, is_daemon) < 0 )
    {
        return 0;  
    }

    /*
    const std::string current_path = CConfigHelper::get_default_config_dir(); 
    const std::string pid = current_path + "/" + CURRENT_SERVER_NAME ".pid";  
    if ( test_running(pid.c_str()) < 0 )
    {
        printf(CURRENT_SERVER_NAME " already run!\n");
        return 1;
    }

    if ( already_running( pid.c_str() ) < 0 )
    {
        printf(CURRENT_SERVER_NAME " already run!\n");
        return 1;
    }*/

    if ( is_daemon )
    {
        init_daemon();
    }

    SetCoreLimit();

    if ( RegisterSomeSignal() < 0 )  return -1;

    if ( StartService() < 0 ) return -1;

    printf("%s start...\n", CURRENT_SERVER_NAME);
    Info("%s start...\n", CURRENT_SERVER_NAME);

    AYServer_Run_loop();

    return 0;
}


static int Usage(int argc, char* argv[], bool& is_daemon)
{
    do
    {
        if (argc <= 1)
        {
            break;
        }

        if (strcmp(argv[1], "--help") == 0)
        {
            printf(CURRENT_SERVER_NAME CURRENT_SERVER_VERSION "\n");
            printf("  --version   Print software version.\n");
            printf("  --compile   Print software compiled time.\n");
            printf("  --debug     Run without daemon.\n");
        }
        else if (strcmp( argv[1], "--version") == 0)
        {
            printf(CURRENT_SERVER_NAME " Software Version:" CURRENT_SERVER_VERSION"\n");     
        }
        else if (strcmp( argv[1], "--compile") == 0)
        {
            printf(CURRENT_SERVER_NAME " Software Compiled Time: %s, %s.\n", __DATE__, __TIME__ );     
        }
        else if (strcmp( argv[1], "--debug") == 0)
        {
            is_daemon = false;
            break;
        }
        else
        {
            printf(CURRENT_SERVER_NAME " Maybe you need try [%s --help]\n", CURRENT_SERVER_NAME);
        }

        return -1;
    }while(false);

    return 0;
}

static void StopProg(int signo)
{
    AYServer_Clear();
    Info("%s stoped...\n", CURRENT_SERVER_NAME);
}

int RegisterSomeSignal()
{
#ifdef _WINDOWS
#else
    struct sigaction act,oact;
    act.sa_handler = StopProg;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    if (sigaction(SIGUSR1,&act,&oact) < 0)
    {
        Error("sigaction SIGUSR1 error!\n");
        return -1;
    }

    act.sa_handler = SIG_IGN;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    if (sigaction(SIGPIPE,&act,&oact) < 0)
    {
        Error("sigaction SIGPIPE error!\n");
        return -1;
    }
#endif
    return 0;
}

static void SetCoreLimit()
{
#ifdef _WINDOWS
#else
    struct rlimit rlim;
    getrlimit(RLIMIT_CORE,&rlim);
    const int core_limit = rlim.rlim_max;
    if (rlim.rlim_cur < core_limit)
    {
        rlim.rlim_cur = core_limit;
        setrlimit(RLIMIT_CORE,&rlim);
    }
#endif
}

static void SetLoglevel(int log_level)
{
    string log_file = CConfigHelper::get_default_config_dir();
    log_file += "/log/";
    log_file += CConfigHelper::get_module_short_name() + ".log";

    setlogfile(log_file);
    setloglevel( (Logger::LogLevel)log_level );
}

static int StartService()
{
    const std::string filename = CConfigHelper::get_default_config_filename();
    uint32 http_port = GetPrivateProfileInt("setting", "http_port", 19010, filename.c_str());
    uint32 serv_port = GetPrivateProfileInt("setting", "serv_port", 19000, filename.c_str());
    uint32 log_level = GetPrivateProfileInt("setting", "log_level", 4, filename.c_str());
    string log_path = CConfigHelper::get_default_config_dir() + "/log";

    if (! StatusServer::ServerInstance()->Start(http_port , serv_port))  // FIXME
    {
        printf("ServerLogical instance start failed!\n");
        return -1;
    }

    SetLoglevel(log_level);

    int ret = AYServer_Init(IServerLogical::GetInstance(), log_path.c_str());
    if (ret< 0)
    {
        printf("AYNet_Init failed, ret=%d!\n", ret);
        return -1;
    }

    ret = AYServer_OpenServ(AYNET_SERV_TYPE_TCP, "0.0.0.0", serv_port);
    if (ret < 0)
    {
        printf("AYNet_OpenServ(%u) failed, ret=%d!\n", serv_port, ret);
        return -1;
    }

    ret = AYServer_OpenServ(AYNET_SERV_TYPE_HTTP, "0.0.0.0", http_port);
    if (ret < 0)
    {
        printf("AYNet_OpenServ(%u) failed, ret=%d!\n", http_port, ret);
        return -1;
    }

    return 0;  //ok
}
