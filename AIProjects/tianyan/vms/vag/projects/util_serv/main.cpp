#include "ServerLogical.h"

#include <sys/time.h>
#include <sys/resource.h>
#include <strings.h>

#include <string>
#include "AYServerApi.h"
#include "DaemonUtil.h"
#include "ConfigHelper.h"

#define SERVER_NAME "util_server"
#define UTIL_SERVER_VERSION "5.17.07.1701"

inline static bool usage(int argc, char* argv[], bool& is_daemon)
{
 do
 {
   if (argc <= 1)  break;
 
   if (strcasecmp(argv[1], "--help") == 0)
   {
     printf(SERVER_NAME UTIL_SERVER_VERSION "\n");
     printf("  --version   Print software version.\n");
     printf("  --compile   Print software compiled time.\n");
     printf("  --debug     Run without daemon.\n");
   }
   else if (strcasecmp( argv[1], "--version") == 0)
   {
     printf(SERVER_NAME " Software Version:" UTIL_SERVER_VERSION "\n");     
   }
   else if (strcasecmp( argv[1], "--compile") == 0)
   {
     printf(SERVER_NAME " Software Compiled Time: %s, %s.\n", __DATE__, __TIME__ );     
   }
   else if (strcasecmp( argv[1], "--debug") == 0)
   {
     is_daemon = false;
     break;
   }
   else
   {
     printf(SERVER_NAME " Maybe you need try [%s --help]\n", SERVER_NAME);     
   }
   
   return true;
 }while(false);

 return false;
}

inline static void stopprog(int signo)
{
  AYServer_Clear();
  Info("util server stoped...\n");
}

inline static int RegisterSomeSignal()
{
  struct sigaction act,oact;
  act.sa_handler = stopprog;
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
  
  return 0;
}

inline static void SetCoreLimit()
{
  struct rlimit rlim;
  getrlimit(RLIMIT_CORE,&rlim);
  const int core_limit = rlim.rlim_max;
  if (rlim.rlim_cur < core_limit)
  {
    rlim.rlim_cur = core_limit;
    setrlimit(RLIMIT_CORE,&rlim);
  }
}

inline static void SetLoglevel(const std::string& current_path)
{
  std::string loglevel = "DEBUG";  
  char loglevel_buf[128] = "";
  GetPrivateProfileString("setting", "log_level",
      loglevel.c_str(),  //default log level
      loglevel_buf,  //new log level
      sizeof(loglevel_buf),
      CConfigHelper::get_default_config_filename().c_str());
  loglevel = loglevel_buf;
  
  setloglevel(loglevel.c_str());
  setlogfile(current_path + "/log/" + SERVER_NAME + ".log");
}

inline static int StartService(const std::string& current_path)
{
  uint32 http_port = GetPrivateProfileInt("setting", "http_port", 
      80,
      CConfigHelper::get_default_config_filename().c_str());
  uint32 server_port = GetPrivateProfileInt("setting", "serv_port", 
      9310,
      CConfigHelper::get_default_config_filename().c_str());
 
  if (!CServerLogical::GetLogical()->Start(0, 0))
  {
    printf("ServerLogical instance start failed!\n");
    return -1;
  }

  const std::string netlib_log_path(current_path+ "/log/"); 
  int ret = AYServer_Init(IServerLogical::GetInstance(), netlib_log_path.c_str());
  if(  ret< 0 )
  {
      printf("AYNet_Init failed, ret=%d!\n", ret);
      return -1;
  }
 
  ret = AYServer_OpenServ(AYNET_SERV_TYPE_TCP, "0.0.0.0", server_port);
  if( ret < 0 )
  {
      printf("AYNet_OpenServ(%u) failed, ret=%d!\n", server_port, ret);
      return -1;
  }
 
  ret = AYServer_OpenServ(AYNET_SERV_TYPE_HTTP, "0.0.0.0", http_port);
  if( ret < 0 )
  {
      printf("AYNet_OpenServ(%u) failed, ret=%d!\n", http_port, ret);
      return -1;
  }

  return 0;  //ok
}

int main(int argc, char *argv[])
{
	setlocale(LC_ALL,"zh_CN.UTF-8");

  bool is_daemon = true;  
  if (usage(argc, argv, is_daemon))
  {
    return 0;  
  }

  const std::string current_path = CConfigHelper::get_default_config_dir(); 
  const std::string pid = current_path + "/" + SERVER_NAME ".pid";  

  if (is_daemon)
  {
    init_daemon();
  }

  if (already_running(pid.c_str()) == -1)
  {
    printf(SERVER_NAME " already run!\n");
    return 1;
  }

  SetLoglevel(current_path);
  
  SetCoreLimit();

  if (RegisterSomeSignal() < 0)  return -1;

  if (StartService(current_path) < 0) return -1;

  Info("util server started...\n");

  pause();
  
  return 0;
}
