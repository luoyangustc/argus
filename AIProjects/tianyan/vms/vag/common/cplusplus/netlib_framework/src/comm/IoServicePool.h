#ifndef __IOSERVICEPOOL_H_
#define __IOSERVICEPOOL_H_

#include <stdio.h>   
#include <cstdlib>   
#include <iostream>   
#include <boost/thread.hpp>   
#include <boost/aligned_storage.hpp>   
#include <boost/array.hpp>   
#include <boost/bind.hpp>   
#include <boost/enable_shared_from_this.hpp>   
#include <boost/noncopyable.hpp>   
#include <boost/shared_ptr.hpp>   
#include <boost/asio.hpp>

class io_service_pool: private boost::noncopyable
{
    typedef boost::shared_ptr<boost::asio::io_service> io_service_ptr;   
    typedef boost::shared_ptr<boost::asio::io_service::work> work_ptr;
public:
    explicit io_service_pool() : m_unNextServiceId(0)
    {
	}

    int Init(std::size_t servs_num, std::size_t threads_per_serv)
    {
        if( (servs_num == 0) || (threads_per_serv == 0) )
        {
            return -1;
        }

        m_unThreadsPerService = threads_per_serv;

        for (std::size_t i = 0; i < servs_num; ++i)
        {
            io_service_ptr io_service(new boost::asio::io_service);
            work_ptr work(new boost::asio::io_service::work(*io_service));
            m_IoServices.push_back(io_service);
            m_Works.push_back(work);
        }

        return 0;
    }

    void Start()
    {
        for (std::size_t i = 0; i < m_IoServices.size(); ++i)
        {
            for(std::size_t j = 0; j < m_unThreadsPerService; ++j)
            {
                m_Threads.create_thread( boost::bind(&boost::asio::io_service::run, m_IoServices[i]) );
            }
		}
    }
  
    void Stop()
    {
        for (std::size_t i = 0; i < m_IoServices.size(); ++i)
		{
            m_IoServices[i]->stop();
		}
    }

    void Join()
    {
        // Wait for all threads in the pool to exit.
        m_Threads.join_all();
    }
  
    // Get an io_service to use.   
    boost::asio::io_service& GetIoService()
    {
        // Use a round-robin scheme to choose the next io_service to use.
        boost::asio::io_service& io_service = *m_IoServices[m_unNextServiceId];        
		++m_unNextServiceId;
        if (m_unNextServiceId == m_IoServices.size())
		{
            m_unNextServiceId = 0;
		}
        return io_service;   
    }
  
private:
    std::vector<io_service_ptr> m_IoServices;	    /// The pool of io_services. 
    std::vector<work_ptr>       m_Works;	    /// The work that keeps the io_services running.   
    boost::thread_group         m_Threads;
    std::size_t                 m_unNextServiceId;
    std::size_t                 m_unThreadsPerService;

};

#endif
