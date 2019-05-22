#include "UDPServer.h"
#include "Log.h"

#ifdef _WINDOWS
#ifdef _DEBUG

#include <stdlib.h>
#include <crtdbg.h>

#ifndef DEBUG_NEW
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif

#endif
#endif //

CUDPServer::CUDPServer( boost::asio::io_service& io_service, ip::udp::endpoint& serv_ep)
    :m_bRunning(false)
    ,m_IoService(io_service)
    ,m_ServSocket(io_service)
    ,m_ServEndpoint(serv_ep)
{
}
	
CUDPServer::~CUDPServer()
{
}

int CUDPServer::Init( IServerLogical* serv_logic )
{
    do 
    {
        if ( !serv_logic )
        {
            break;
        }

        m_pServiceLogic = serv_logic;
        m_unWorkThreadCnt = 8;//threads_num;

        m_bRunning = false;

        DEBUG_LOG( "CUDPServer::Init--->ServPort(%u), Success.", m_ServEndpoint.port() );

        return 0;

    } while (0);       

    return -1;
}
	
int CUDPServer::Start()
{
    do 
    {
        m_ServSocket.open( m_ServEndpoint.protocol() );
        m_ServSocket.set_option(ip::udp::socket::reuse_address(true));
        m_ServSocket.set_option(ip::udp::socket::send_buffer_size(32*1024));
        m_ServSocket.set_option(ip::udp::socket::receive_buffer_size(32*1024));
        m_ServSocket.bind(m_ServEndpoint);

        m_bRunning = true;

        for( uint32 i=0; i < m_unWorkThreadCnt; ++i )
        {
            m_WorkThreadGroup.create_thread(boost::bind(&CUDPServer::WorkRunProc, shared_from_this(), i));
        }

        m_ReadThread.reset( new boost::thread( boost::bind( &CUDPServer::ReadRunProc, shared_from_this() ) ) );

        //StartAsyncRead();

        DEBUG_LOG( "CUDPServer::start--->ServPort(%u), success.", m_ServEndpoint.port() );

        return 0;

    } while (0);

    return -1;
}

int CUDPServer::Stop()
{
    do 
    {
        m_bRunning = false;

        m_WorkCond.notify_all();

        DEBUG_LOG( "CUDPServer::stop--->ServPort(%u).", m_ServEndpoint.port() );

        return 0;

    } while (0);

    return -1;
}

void CUDPServer::Update()
{
    
}

bool CUDPServer::WorkRunProc(uint32 thread_idx)
{
    DEBUG_LOG( "CUDPServer::WorkRunProc--->ServPort(%u), ThreadIdx(%u).", m_ServEndpoint.port(), thread_idx );

    while (m_bRunning)
    {
        SUDPMessageInfo udp_msg_info;
        if ( PopQueue(udp_msg_info) )
        {
            CDataStream recvds(udp_msg_info.data.get_buffer(), udp_msg_info.data.data_size_);
            char send_buff[2*1024];
            CDataStream sendds(send_buff, sizeof(send_buff));

            if( m_pServiceLogic->OnUDPMessage(udp_msg_info.hi_remote, recvds, sendds, thread_idx ,0) < 0 )
            {
                continue;
            }
            in_addr sin_addr;
            sin_addr = *((in_addr*)&udp_msg_info.hi_remote.IP);
            string a;
            ip::udp::endpoint remote_ep(ip::address_v4::from_string(inet_ntoa(*((in_addr*)&udp_msg_info.hi_remote.IP))), udp_msg_info.hi_remote.Port);
            
            if(sendds.size())
            {
                m_ServSocket.send_to(buffer(sendds.getbuffer(), sendds.size()), remote_ep);
            }
        }
    }

    return false;
}

bool CUDPServer::ReadRunProc()
{
    #define MAX_UDP_MSG_LEN (16*1024 + 2*1024)

    DEBUG_LOG( "CUDPServer::ReadRunProc--->Enter read thread function, ServPort(%u).", m_ServEndpoint.port() );

    while (m_bRunning)
    {
        boost::system::error_code error;
        boost::asio::ip::udp::endpoint client_endpoint;
        SDataBuff recv_buff;
        if( !recv_buff.resize(MAX_UDP_MSG_LEN) )
        {
            boost::this_thread::sleep( boost::posix_time::millisec(100));
            continue;
        }

        recv_buff.data_size_ = m_ServSocket.receive_from( buffer(recv_buff.pbuff_.get(), recv_buff.buff_size_), client_endpoint, 0, error );            
        if ( error )
        {
            ERROR_LOG( "CUDPServer::ReadRunProc--->ServPort(%u), read error(%s).", m_ServEndpoint.port(),error.message().c_str() );
            boost::this_thread::sleep(boost::posix_time::millisec(100));
            continue;
        }
        if ( recv_buff.data_size_ )
        {
            SUDPMessageInfo udp_msg_info;
            udp_msg_info.hi_remote.IP = inet_addr(client_endpoint.address().to_string().c_str());
            udp_msg_info.hi_remote.Port = client_endpoint.port();
            udp_msg_info.data = recv_buff;
            PushQueue(udp_msg_info);
        }
    }

    return false;
}

bool CUDPServer::PushQueue(IN const SUDPMessageInfo& msg_info)
{
    {
        boost::mutex::scoped_lock lock(m_WorkMutex);
        m_WorkQueue.push(msg_info);
        m_WorkCond.notify_one();
    }

    if( m_WorkQueue.size() > m_unWorkThreadCnt*20 )
    {
        DEBUG_LOG( "CUDPServer::PushQueue--->ServPort(%u), msg_info(%s, %u), queue_size(%u).", 
            m_ServEndpoint.port(), 
            msg_info.hi_remote.GetNodeString().c_str(),
            msg_info.data.data_size_,
            m_WorkQueue.size() );

        boost::mutex::scoped_lock read_lock(m_ReadMutex);
        m_ReadCond.wait(read_lock);
    }    

    return true;
}

bool CUDPServer::PopQueue(OUT SUDPMessageInfo& msg_info)
{
    do 
    {
        {
            boost::mutex::scoped_lock lock(m_WorkMutex);

            if ( m_WorkQueue.empty() )
            {
                while ( m_WorkQueue.empty() )
                {
                    WARN_LOG( "CUDPServer::PopQueue--->ServPort(%u), enter wait.", m_ServEndpoint.port() );
                    m_WorkCond.wait(lock);
                }
            }

            if( !m_bRunning )
            {
                WARN_LOG( "CUDPServer::PopQueue--->ServPort(%u), server has stoped!", m_ServEndpoint.port() );
                break;
            }

            msg_info =  m_WorkQueue.front();
            m_WorkQueue.pop();

            m_ReadCond.notify_one();
        }

        /*
        DEBUG_LOG( "CUDPServer::PopQueue--->ServPort(%u), msg_info(%s, %u), queue_size(%u).", 
            m_ServEndpoint.port(), 
            msg_info.hi_remote.GetNodeString().c_str(),
            msg_info.data.len,
            m_WorkQueue.size());
        */

        return true;

    } while (0);
    
    return false;
}

string CUDPServer::GetServerName()
{
    string name = "UDP-" + m_ServEndpoint.address().to_string();
    return name;
}

std::ostringstream& CUDPServer::DumpInfo(std::ostringstream& oss)
{
    oss << "{";
    oss << "}";
    return oss;
}
