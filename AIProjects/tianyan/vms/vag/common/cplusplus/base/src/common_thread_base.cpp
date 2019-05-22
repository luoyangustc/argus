#include "common_thread_base.h"

CCommon_Thread_Base::CCommon_Thread_Base()
    :m_bRunning(false)
    ,m_unRunCycle(100) //default, 100us
{ 
}

CCommon_Thread_Base::~CCommon_Thread_Base()
{
}
#if 1
bool CCommon_Thread_Base::Start(en_thread_mode mode)
{
    m_bRunning = true;
    m_pThread.reset( new boost::thread( boost::bind( &CCommon_Thread_Base::MainLoop, shared_from_this() ) ) );
    if(!m_pThread)
    {
        m_bRunning = false;
        return false;
    }
    m_nMode = mode;
    if(m_nMode == en_mode_detach)
    {
        m_pThread->detach();
    }
    return true;
}

bool CCommon_Thread_Base::Stop()
{
    m_bRunning = false;

    return true;
}

bool CCommon_Thread_Base::Join()
{
    if(m_bRunning && (m_nMode == en_mode_join))
    {
        m_pThread->join();
    }

    return true;
}

bool CCommon_Thread_Base::Join(unsigned int wait_usec)
{
    bool ret = true;
    if(m_bRunning && (m_nMode == en_mode_join))
    {
        ret = m_pThread->timed_join(boost::posix_time::microseconds(wait_usec));
    }

    return ret;
}
bool CCommon_Thread_Base::SetRunCycle(unsigned int usec)
{
    m_unRunCycle = usec;
    return true;
}

void CCommon_Thread_Base::MainLoop()
{
    while(m_bRunning)
    {
        if( Run() < 0 )
        {
            break;
        }

        boost::this_thread::sleep(boost::posix_time::microseconds(m_unRunCycle));
    }
}

#endif