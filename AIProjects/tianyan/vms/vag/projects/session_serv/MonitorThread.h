#ifndef __MONITOR_THREAD_H__
#define __MONITOR_THREAD_H__

#include <boost/thread/recursive_mutex.hpp>
#include <boost/shared_ptr.hpp>
#include "tick.h"
#include "common_thread_base.h"

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