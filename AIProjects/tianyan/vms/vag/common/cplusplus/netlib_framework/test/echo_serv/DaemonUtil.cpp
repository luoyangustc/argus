#include "DaemonUtil.h"

#include "typedefine.h"
#include "typedef_win.h"

#define DEBUG_PRINTF printf

#ifndef _WINDOWS
#define MAXFILE 65535
void init_daemon(void)
{
    int pid;
    int i;
    
    DEBUG_PRINTF("init_daemon--enter\n");

    signal(SIGTTOU, SIG_IGN);
    signal(SIGTTIN, SIG_IGN);
    signal(SIGTSTP, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    struct rlimit   rl;                       //获取进程资源西限制  
    if (getrlimit(RLIMIT_NOFILE, &rl) < 0)    //获取进程最多文件数  
    {
        DEBUG_PRINTF(":can't get file limit");
    }      
    
    if(pid=fork())
    {
        exit(0);//是父进程，结束父进程
    }
    else if(pid< 0)
    {
        DEBUG_PRINTF("init_daemon--fork failed, pid=%d\n", pid);
        exit(1);//fork失败，退出
    }
    
    DEBUG_PRINTF("init_daemon--first child processid=%d\n", getpid());
    //是第一子进程，后台继续执行
    setsid();//调用成功后，第一子进程成为新的会话组长和新的进程组长，并与原来的登录会话和进程组脱离
    
#if 1
    //会话组长可以打开控制终端，需要进一步脱离会话组长
    if(pid=fork())
    {
        exit(0);//是第一子进程，结束第一子进程
    }
    else if(pid< 0)
    {
        DEBUG_PRINTF("init_daemon--fork failed, pid=%d\n", pid);
        exit(1);//fork失败，退出
    } 
#endif
    DEBUG_PRINTF("init_daemon--second child processid=%d\n", getpid());
    //是第二子进程，继续
    //第二子进程不再是会话组长

    //chdir("/tmp");  //改变工作目录到/tmp

    /* 关闭打开的文件描述符*/  
    if (rl.rlim_max == RLIM_INFINITY)     //RLIM_INFINITY是一个无穷量的限制        
    {
        rl.rlim_max = 1024;  
    }
    for (i = 0; i < rl.rlim_max; i++)  
    {
        close(i);
    }
    
    //open("/dev/null", O_RDONLY);  
    //open("/dev/null", O_RDWR);  
    //open("/dev/null", O_RDWR);
    
    umask(0);//重设文件创建掩模

    signal(SIGCHLD,SIG_IGN);
    
    DEBUG_PRINTF("init_daemon--success!\n");
}
int lockfile(int)
{
	return 0;
}
int already_running(const char szLockFile[])
{
	return 0;
}
int test_running(const char szLockFile[])
{
	return 0;
}
int test_local_process_running(const char szLockFile[])
{
	return 0;
}
#else
void init_daemon(void)
{
}

int lockfile(int)
{
	return 0;
}
int already_running(const char szLockFile[])
{
	return 0;
}
int test_running(const char szLockFile[])
{
	return 0;
}
int test_local_process_running(const char szLockFile[])
{
	return 0;
}

#endif
