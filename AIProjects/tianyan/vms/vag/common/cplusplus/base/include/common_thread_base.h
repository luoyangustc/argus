
#ifndef __THREAD_BASE__
#define __THREAD_BASE__

#include <stdint.h>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread.hpp>
#include <boost/shared_array.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#define BOOST_DATE_TIME_SOURCE
#include <string>

class CCommon_Thread_Base:public boost::enable_shared_from_this<CCommon_Thread_Base>
{
public:
    enum en_thread_mode
    {
        en_mode_join = 0,
        en_mode_detach = 1,
    };
public:
    CCommon_Thread_Base();
    virtual ~CCommon_Thread_Base();
    void MainLoop();
    bool Start(en_thread_mode mode = en_mode_join);
    bool Stop();
    bool Join();
    bool Join(unsigned int wait_usec);
    bool SetRunCycle(unsigned int usec);
public:
    virtual bool Run() = 0;
private:
    boost::shared_ptr<boost::thread> m_pThread;
    en_thread_mode m_nMode;
    boost::atomic_bool m_bRunning;
    unsigned int m_unRunCycle; //usec
};

#endif /* defined(__THREAD_BASE__) */
