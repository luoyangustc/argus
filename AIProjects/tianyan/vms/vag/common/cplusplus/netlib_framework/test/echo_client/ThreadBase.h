
#ifndef __THREAD_BASE__
#define __THREAD_BASE__

#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_array.hpp>
#include <boost/array.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#define BOOST_DATE_TIME_SOURCE

using namespace std;

class CThreadBase:public boost::enable_shared_from_this<CThreadBase>
{
public:
    CThreadBase();
    virtual ~CThreadBase();
    bool Start();
    bool Stop();
    bool Join();
    bool Join(unsigned int wait_usec);
public:
    virtual void Run() = 0;
private:
    boost::atomic_bool m_bRunning;
    boost::shared_ptr<boost::thread> m_pThread;
};

#endif /* defined(__THREAD_BASE__) */
