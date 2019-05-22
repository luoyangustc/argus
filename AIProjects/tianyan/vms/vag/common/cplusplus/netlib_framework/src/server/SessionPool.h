#ifndef __SESSIONPOOL_H__
#define __SESSIONPOOL_H__

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include "Common.h"
#include "Log.h"

using namespace std;
using boost::asio::ip::tcp;

class io_service_pool;
template<typename SESSION_T>
class CSessionPool
	: public boost::enable_shared_from_this<CSessionPool<SESSION_T> >//, private boost::noncopyable
{
public:
	typedef boost::shared_ptr<SESSION_T> SESSION_PTR;
	CSessionPool(boost::asio::io_service& io_svc)
		:m_IoService(io_svc)
	{}
	~CSessionPool(){}
	
	int Init(std::size_t session_pool_size=100)
	{
		for(uint32 i=0; i<session_pool_size; i++)
		{
			SESSION_T *session_handle = new SESSION_T(m_IoService);
			SESSION_PTR session_ptr(session_handle, boost::bind(&CSessionPool::PushSession, this->shared_from_this(), _1) );
			if(session_ptr.get() == 0)
			{
				return -1;
			}
            //DEBUG_LOG("CSessionPool::Init-->Session(%p,%d)\n", session_ptr.get(), session_ptr.use_count());
			m_SessionList.push(session_ptr);
            //DEBUG_LOG("CSessionPool::Init02-->Session(%p,%d)\n", session_ptr.get(), session_ptr.use_count());
		}
		return 0;
	}
	
	void PushSession(SESSION_T *pSession)
	{
		SESSION_PTR session_ptr(pSession, boost::bind(&CSessionPool::PushSession,
			this->shared_from_this(),
			_1));
        
        session_ptr->Reset();

        {
            boost::asio::detail::mutex::scoped_lock lock(m_SessionListLock);
            m_SessionList.push(session_ptr);
        }
        DEBUG_LOG("CSessionPool::PushSession-->Session(%p,%d)\n", session_ptr.get(), session_ptr.use_count());
		
	}

	SESSION_PTR GetSession()
	{
		SESSION_PTR session;

		do 
		{
            boost::asio::detail::mutex::scoped_lock lock(m_SessionListLock);
			if(m_SessionList.empty())
			{
				SESSION_T *session_handle = new SESSION_T(m_IoService);
				SESSION_PTR session_ptr(session_handle, boost::bind(&CSessionPool::PushSession, this->shared_from_this(), _1) );
				if(session_ptr.get() == 0)
				{
					break;
				}
				m_SessionList.push(session_ptr);
			}

			if(m_SessionList.size() > 0)
			{
				session = m_SessionList.front();
				m_SessionList.pop();
			}
		} while (0);

        DEBUG_LOG("CSessionPool::GetSession-->Session(%p,%d)\n", session.get(), session.use_count());
		return session;
	}

    uint32 GetSessionPoolSize()
    {
        boost::asio::detail::mutex::scoped_lock lock(m_SessionListLock);
        return m_SessionList.size();
    }

    std::ostringstream& DumpInfo(std::ostringstream& oss)
    {
        oss << "{";
        oss << "\"SessionPoolSize\":";
        oss << GetSessionPoolSize();
        oss << "}";
    }
private:
	boost::asio::io_service&    m_IoService;
	std::queue<SESSION_PTR>	    m_SessionList;
	boost::asio::detail::mutex 	m_SessionListLock;
};

#endif
