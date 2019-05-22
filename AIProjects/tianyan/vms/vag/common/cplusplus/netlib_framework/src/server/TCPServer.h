#ifndef __TCP_SERVER_H_
#define __TCP_SERVER_H_

#include <map>
#include <queue>
#include <set>
#include "Common.h" 
#include "typedefine.h"
#include "Log.h"
#include "ServerBase.h"
//#include "SessionPool.h"
#include "HandlerAllocator.h"
#include "IoServicePool.h"

using namespace std;
using namespace boost::asio;

//extern uint32 gTcpSessionCnt;

struct SSessionRunStatus
{
    bool    m_bRunning;
    bool    waiting_;
    bool    last_run_tick_;
};

template<typename SESSION_T>
class CTCPServer:public IServerBase
{
public:
    typedef boost::shared_ptr<SESSION_T>                SESSION_PTR;
    //typedef CSessionPool<SESSION_T>                     SESSION_POOL_T;
    //typedef boost::shared_ptr<CSessionPool<SESSION_T> > SESSION_POOL_PTR;
public:
    CTCPServer(boost::asio::io_service& io_serv, io_service_pool& task_srv_pool, ip::tcp::endpoint& serv_ep)
        :m_bRunning(false)
        ,m_ServEndpoint(serv_ep)
        ,m_IoService(io_serv)
        ,m_TaskServicePool(task_srv_pool)
        ,m_spNewSession()
        ,m_unTaskCnt(0)
        ,m_unTaskDoingCnt(0)
    {
    }

    ~CTCPServer()
    {
    }

    int StartAccept()
    {
        m_spNewSession.reset(new SESSION_T(m_IoService)); 
        //m_spNewSession = m_spSessionPool->GetSession();
        if(m_spNewSession.get() == 0)
        {
            ERROR_LOG( "Get Session failed!" );
            return -1;
        }
        //m_Acceptor.async_accept(*new_session->Socket(),
        //    make_custom_alloc_handler( m_Allocator, boost::bind(&CTCPServer::HandleAccept, this, new_session, boost::asio::placeholders::error)) );
        m_spAcceptor->async_accept(m_spNewSession->Socket(),
            boost::bind(&CTCPServer::HandleAccept, this, boost::asio::placeholders::error) );

        return 0;
    }

    int Init( IServerLogical* serv_logic )
    {
        do
        {
            if ( !serv_logic )
            {
                ERROR_LOG( "serv_logic is nil." );
                break;
            }

            m_spAcceptor.reset( new boost::asio::ip::tcp::acceptor(m_IoService) );
            if( !m_spAcceptor.get() )
            {
                ERROR_LOG( "Create Acceptor failed!" );
                break;
            }

            m_spSignals.reset(new boost::asio::signal_set(m_IoService));
            if( !m_spSignals.get() )
            {
                ERROR_LOG( "Create Sigal Set failed!" );
                break;
            }

            /*
            m_spSessionPool = SESSION_POOL_PTR( new SESSION_POOL_T(m_IoService) );
            if( !m_spSessionPool.get() )
            {
                ERROR_LOG( "CTCPServer::Init-->Create Session Pool failed!" );
                break;
            }

            m_unSessionPoolSize = 100;

            if ( m_spSessionPool->Init(m_unSessionPoolSize) != 0 )
            {
                ERROR_LOG( "CTCPServer::Init-->Init Session Pool failed!" );
                break;
            }*/

            m_pServiceLogic = serv_logic;

            m_bRunning = false;

            DEBUG_LOG( "Success." );

            return 0;

        } while (0);

        return -1;
    }

    int Start()
    {
        do
        {
            m_bRunning = true;

            //m_IoServicePool.Start();
            //m_TaskServicePool.Start();

            m_spSignals->add(SIGINT);
            m_spSignals->add(SIGTERM);
            #if defined(SIGQUIT)
            m_spSignals->add(SIGQUIT);
            #endif
            m_spSignals->async_wait(boost::bind(&CTCPServer::HandleStop, this));

            m_spAcceptor->open(m_ServEndpoint.protocol());
            m_spAcceptor->set_option(ip::tcp::acceptor::reuse_address(true));
            m_spAcceptor->bind(m_ServEndpoint);
            //m_Acceptor.listen(ip::tcp::acceptor::max_connections);
            m_spAcceptor->listen(2*1024);

            return StartAccept();

        } while (0);

        return -1;
    }

    int Stop()
    {
        do 
        {
            m_bRunning = false;

            //m_IoServicePool.Stop();
            //m_TaskServicePool.Stop();

            //m_IoServicePool.Join();
            //m_TaskServicePool.Join();

            DEBUG_LOG( "CTCPServer::stop-->Success." );

            return 0;

        } while (0);

        return -1;
    }

    void Update()
    {
        do 
        {
           CheckTimeout();
        } while (0);
    }

    void AddTask(SESSION_PTR session)
    {
        /*
        ++m_unTaskCnt;
        if(m_unTaskCnt>50)
        {
            WARN_LOG( "add task(%p), task_cnt=%u.", session.get(), (unsigned int)m_unTaskCnt );
        }*/

        {
            boost::lock_guard<boost::recursive_mutex> lock(m_TaskQueueLock);
            m_TaskServicePool.GetIoService().post( boost::bind(&CTCPServer::HandleTask, this, session) );
        }
    }

    int32 OnTCPAccepted(SESSION_PTR session,CHostInfo& hiRemote,CDataStream& sendds)
    {
        DEBUG_LOG("CTCPServer::OnTCPAccepted-->Session(%p,%d).", session.get(), session.use_count());
        return m_pServiceLogic->OnTCPAccepted( session.get(), hiRemote, sendds );
    }

    int32 OnTCPMessage( SESSION_PTR session, CHostInfo& hiRemote, uint32 msg_id, CDataStream& recvds, CDataStream& sendds)
    {
        //DEBUG_LOG("CTCPServer::OnTCPMessage-->Session(%p,%d).", session.get(), session.use_count());
        return m_pServiceLogic->OnTCPMessage( session.get(), hiRemote, msg_id, recvds, sendds );
    }

    int32 OnHttpClientRequest( SESSION_PTR session, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pRes )
    {
        //DEBUG_LOG("CTCPServer::OnHttpClientRequest-->Session(%p,%d).", session.get(), session.use_count());
        return m_pServiceLogic->OnHttpClientRequest(session.get(), hiRemote, pReq, pRes);
    }

    
    int32 OnTCPClosed(SESSION_PTR session,CHostInfo& hiRemote)
    {
        m_pServiceLogic->OnTCPClosed( session.get(), hiRemote );

        //boost::lock_guard<boost::recursive_mutex> lock(m_SessionSetLock);
        //m_SessionSet.erase(session);
        //DEBUG_LOG("CTCPServer::OnTCPClosed-->Session(%p,%d).", session.get(), session.use_count());
        DEBUG_LOG("Server(%p,%u), session(%p)", this, m_ServEndpoint.port(), session.get());
        return 0;
    }

    uint32 GetConnectedSessionSize()
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_SessionSetLock);
        return m_SessionSet.size();
    }

    string GetServerName()
    {
        string name = "TCP-" + m_ServEndpoint.address().to_string();
        return name;
    }

    std::ostringstream& DumpInfo(std::ostringstream& oss)
    {
        oss << "{";
        oss << "\"ListenPort\":" << m_ServEndpoint.port();

        oss << ",";
        oss << "\"ConnectedSessionSize\":" << GetConnectedSessionSize();

        oss << ",";
        oss << "\"TaskCnt\":" << m_unTaskCnt;

        //oss << ",";
        //oss << "\"SessionCnt\":" << gTcpSessionCnt;
        //oss << "}";

        return oss;
    }

private:
    void HandleAccept(const boost::system::error_code& error)   
    {
        DEBUG_LOG("Session(%p,%d).", 
            m_spNewSession.get(),
            m_spNewSession.use_count());
        do
        {
            if ( error )
            {
                ERROR_LOG(
                    "Server(%p), error(%d,%s).", 
                    this, 
                    error.value(),
                    error.message().c_str()
                    );
                //Stop();
                break;
            }

            if( !m_bRunning || !m_spNewSession.get() || !m_spNewSession->Socket().is_open())
            {
                ERROR_LOG("Server(%p), sys_error(%d,%p, %d).", 
                    this, 
                    m_spNewSession.get(),
                    m_spNewSession->Socket().is_open());
                Stop();
                break;
            }
            
            if( m_spNewSession->Init(this) == 0 )
            {
                boost::lock_guard<boost::recursive_mutex> lock(m_SessionSetLock);
                m_SessionSet.insert(m_spNewSession);
            }

        }while(0);

        if( StartAccept() < 0 )
        {
            ERROR_LOG("Server(%p), StartAccept failed!", this);
            Stop();
        }
    }

    void HandleTask(SESSION_PTR session)
    {
        {
            //++m_unTaskDoingCnt;
            //uint32 tasking_cnt = (uint32)m_unTaskDoingCnt;
            //if(tasking_cnt > 20)
            //{
            //    WARN_LOG( "doing task(%p), task_doing_cnt=%u.", session.get(), tasking_cnt );
            //}
        }

        if( !session.get() )
        {
            WARN_LOG("Session is nil!");
            return;
        }

        session->TaskMain();

        {
            //boost::lock_guard<boost::recursive_mutex> lock(m_TaskQueueLock);
            //--m_unTaskCnt;
            //--m_unTaskDoingCnt;
        }
    }

    void HandleStop()
    {
        DEBUG_LOG("Server(%s,%d).", 
            m_ServEndpoint.address().to_string().c_str(),
            m_ServEndpoint.port());

        m_spAcceptor->close();

        Stop();
    }

    void CheckTimeout()
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_SessionSetLock);
        typename std::set< boost::shared_ptr<SESSION_T> >::iterator it = m_SessionSet.begin();
        while( it != m_SessionSet.end() )
        {
            SESSION_PTR pSession = *it;
            if( !pSession->IsAlive() )
            {
                ERROR_LOG("CTCPServer::CheckTimeout-->Server(%p,%u), remove session(%p)", 
                    this, m_ServEndpoint.port(), pSession.get());
                m_SessionSet.erase(it++);
                
            }
            else
            {
                ++it;
            }
        }
    }

private:
    bool                            m_bRunning;
    IServerLogical*                 m_pServiceLogic;
    boost::asio::ip::tcp::endpoint  m_ServEndpoint;
    io_service&                     m_IoService;
    io_service_pool&                m_TaskServicePool;
    Acceptor_ptr                    m_spAcceptor;
    SignalSet_ptr                   m_spSignals;
    handler_allocator               m_Allocator;
    std::size_t                     m_unSessionPoolSize;
    boost::atomic_uint32_t          m_unTaskCnt;
    boost::atomic_uint32_t          m_unTaskDoingCnt;
    //unsigned int                    m_unTaskCnt;
    //unsigned int                    m_unTaskDoingCnt;
    boost::atomic_uint32_t          m_unDelTaskCnt;
    boost::recursive_mutex          m_TaskQueueLock;

    std::set<SESSION_PTR>           m_SessionSet;
    boost::recursive_mutex          m_SessionSetLock;

    SESSION_PTR                     m_spNewSession;
    //SESSION_POOL_PTR                m_spSessionPool;
};

#endif