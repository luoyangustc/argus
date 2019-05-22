#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <boost/thread/lock_guard.hpp>
#include "LFile.h"
#include "logging_posix.h"
#include "MonitorThread.h"

CSysMonitorThread::CSysMonitorThread() {
    last_active_tick_ = get_current_tick();
    SetRunCycle(2*1000*1000);
}
CSysMonitorThread::~CSysMonitorThread() {
    Stop();
}

void CSysMonitorThread::UpdateActiveTick() {
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    last_active_tick_ = get_current_tick();
}

bool CSysMonitorThread::Run() {

    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    tick_t current_tick = get_current_tick();
    if ( current_tick - last_active_tick_ > 30*1000 )
    {
        Fatal("[CCheckActiveThread] Exit current_tick(%lld), last_active_tick(%lld)", current_tick, last_active_tick_);
        //程序死锁退出前打印线程堆栈
        string ofn = CLFile::GetModuleDirectory() + "/deadlock_stack.txt";

        time_t now = time(NULL);
        struct tm ntm;
        localtime_r(&now, &ntm);
        char time_buf[256] = {0};
        sprintf(time_buf, "%d-%02d-%02d %02d:%02d:%02d", 
            ntm.tm_year + 1900, ntm.tm_mon + 1, ntm.tm_mday,
            ntm.tm_hour, ntm.tm_min, ntm.tm_sec);

        char cmd[512];
        snprintf(cmd,sizeof(cmd),"echo -e \"----------------%s deadlock stack:\r\n\" >> %s", time_buf, ofn.c_str());
        std::system(cmd);

        snprintf(cmd,sizeof(cmd),"pstack %d >> %s", getpid(), ofn.c_str());
        std::system(cmd);

        snprintf(cmd,sizeof(cmd),"echo -e \"\r\n\r\n\r\n\r\n\r\n\r\n\" >> %s", ofn.c_str());
        std::system(cmd);

        exit(0);
        return false;
    }

    return true;
}
