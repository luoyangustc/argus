#ifndef __MONITOR_THREAD_H__
#define __MONITOR_THREAD_H__

#include "CommonInc.h"

class CSysMonitorThread : public CCommon_Thread_Base
{
public:
    CSysMonitorThread();
    ~CSysMonitorThread();

    void UpdateActiveTick();

    virtual bool Run();

private:
    boost::recursive_mutex lock_;
    tick_t last_active_tick_;
};

typedef boost::shared_ptr<CSysMonitorThread> CSysMonitorThread_ptr;

#endif