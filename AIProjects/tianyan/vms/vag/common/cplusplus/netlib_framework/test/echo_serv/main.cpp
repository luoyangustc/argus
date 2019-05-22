
#ifndef _WINDOWS
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#else
//#include <windows.h>
#endif

#include <string.h>
#include <string>
#include "DaemonUtil.h"
#include "Server/ServerApi.h"
#include "ServerLogical.h"


std::string g_prog_name = "test_serv";

void stopprog(int signo);
void usage()
{
    printf( "%s\n", g_prog_name.c_str() );
    exit(0);
}

int main(int argc, char *argv[])
{
	string sVerMagic = "--version";
	string sVersion = "1.0";

	bool is_daemon = true;
    uint32 work_port = 0;
    uint32 serv_port = 8000;
	do
	{
	    if( argc > 1 )
	    {
            if( strcmp( argv[1],sVerMagic.c_str() )==0 )
            {
                printf("%s %s\n", g_prog_name.c_str(), sVersion.c_str());
                return 0;
            }
            else if( strcmp( argv[1],"--debug" )==0 )
            {
                is_daemon = false;
            }
            else if( argc == 2 )
            {
                serv_port = (uint32)atoi(argv[1]);
                if( serv_port <= 0 )
                {
                    printf("parser listen port error, %s\n", argv[1]);
                    return 0;
                }
                work_port = serv_port + 10;
                printf("serv_port=%u, %s\n", serv_port);
            }
            else
            {
                printf("%s %s\n", g_prog_name.c_str(), sVersion.c_str());  
                return 0;
            }
        }
	}while(false);

#ifndef _WINDOWS

    if( is_daemon )
    {
        init_daemon();
    }

    struct rlimit rlim;
    getrlimit(RLIMIT_CORE,&rlim);
    const int core_limit = rlim.rlim_max;
    if( rlim.rlim_cur < core_limit )
    {
        rlim.rlim_cur = core_limit;
        setrlimit(RLIMIT_CORE,&rlim);
    }

    struct sigaction act,oact;
    act.sa_handler = stopprog;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    if( sigaction(SIGUSR1,&act,&oact) < 0 )
    {
        printf("sigaction faild!\n");
        return -1;
    }
#endif

	if (work_port == 0 )
	{
		work_port = serv_port + 10;
	}
    
    ServInitCfg cfg;
    cfg.service_logic_ptr = CServerLogical::GetLogical();
    cfg.log_path = "";
    ServInfo serv;
    serv.type = EN_SERV_TYPE_UDP;
    serv.ip = "0.0.0.0";
    serv.port = serv_port;
    cfg.serv_list.push_back(serv);
    
    serv.type = EN_SERV_TYPE_TCP;
    serv.ip = "0.0.0.0";
    serv.port = serv_port;
    cfg.serv_list.push_back(serv);
    
    serv.type = EN_SERV_TYPE_HTTP;
    serv.ip = "0.0.0.0";
    serv.port = work_port;
    cfg.serv_list.push_back(serv);
    
	if( !Serv_Start(cfg) )
    {
        return -1;
    }
    
	Serv_Run_loop();

	return 0;
}

void stopprog(int signo)
{
    Serv_Stop();
}

