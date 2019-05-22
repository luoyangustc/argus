//#include "stdafx.h"

#include "ThreadBase.h"

CThreadBase::CThreadBase():m_bRunning(false)
{ 
}

CThreadBase::~CThreadBase()
{
}

bool CThreadBase::Start()
{
    m_bRunning = true;
    m_pThread.reset( new boost::thread( boost::bind( &CThreadBase::Run, shared_from_this() ) ) );
    if(!m_pThread)
    {
        m_bRunning = false;
        return false;
    }

    return true;
}

bool CThreadBase::Stop()
{
    m_bRunning = false;

    return true;
}

bool CThreadBase::Join()
{
    if(m_bRunning)
    {
        m_pThread->join();
    }

    return true;
}

bool CThreadBase::Join(unsigned int wait_usec)
{
    bool ret = true;
    if(m_bRunning)
    {
        ret = m_pThread->timed_join(boost::posix_time::microseconds(wait_usec));
    }

    return ret;
}