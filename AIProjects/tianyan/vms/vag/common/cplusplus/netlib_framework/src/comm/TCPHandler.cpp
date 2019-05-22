#include "TCPHandler.h"
#include "Log.h"
#include "HandlerAllocator.h"

CTCPHandler::CTCPHandler( boost::asio::io_service& io_service )
    :m_IoService( io_service )
    ,m_Strand( io_service )
    ,m_Sock( io_service )
    ,m_usRemotePort( 0 )
    ,m_enTcpStatus(en_tcp_sts_init)
    ,m_enTcpMode(en_tcp_on_async)
    ,m_bRunning( false )
    ,m_bSending(false)
    ,m_unLastRwTime( 0 )

{
}

CTCPHandler::~CTCPHandler()
{
}

bool CTCPHandler::IsOpen()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    return m_Sock.is_open() && (m_enTcpStatus == en_tcp_sts_connected);
}

bool CTCPHandler::IsAlive()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if ( m_enTcpStatus == en_tcp_sts_disconnect 
        || m_enTcpStatus == en_tcp_sts_connect_failed 
        || m_enTcpStatus == en_tcp_sts_closed )
    {
        WARN_LOG( "socke error, Session(%p), LocalAddr(%s), RemoteAddr(%s), tcp_status(%d), LastRwTime(%llu).", 
            this, GetLocalAddr().c_str(), GetRemoteAddr().c_str(), m_enTcpStatus, m_unLastRwTime );
        return false;
    }

    time_t cur_time = time(NULL);
    if( m_unLastRwTime && ((cur_time - m_unLastRwTime) > 120) ) //2 min
    {
        WARN_LOG( "timeout, Session(%p), LocalAddr(%s),RemoteAddr(%s), LastRwTime(%llu).", 
            this, GetLocalAddr().c_str(), GetRemoteAddr().c_str(), m_unLastRwTime );

        Close();

        return false;
    }
    return true;
}

void CTCPHandler::SetMode(EN_TCP_MODE mode)
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    m_enTcpMode = mode;
}

int CTCPHandler::Open()
{
    int ret = 0;

    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
        if( !m_Sock.is_open() )
        {
            ret = -1;
            break;
        }

        boost::system::error_code ec;
        boost::asio::ip::tcp::endpoint remote_ep = m_Sock.remote_endpoint(ec);
        if (ec)
        {
            ERROR_LOG( "Session(%p),get remote address failed, err=%d(%s)!", this, ec.value(), ec.message().c_str());
            ret = -2;
            break;
        }

        ec.clear();
        boost::asio::ip::tcp::endpoint local_ep = m_Sock.local_endpoint(ec);
        if (ec)
        {
            ERROR_LOG( "Session(%p),get local address failed, err=%d(%s)!", this, ec.value(), ec.message().c_str());
            ret = -3;
            break;
        }

        m_strRemoteIP = remote_ep.address().to_string();
        m_usRemotePort = remote_ep.port();
        m_strLocalIP = local_ep.address().to_string();
        m_usLocalPort = local_ep.port();
        m_Sock.set_option(ip::tcp::socket::send_buffer_size(32*1024));
        m_Sock.set_option(ip::tcp::socket::receive_buffer_size(32*1024));
        m_unLastRwTime = time(NULL);
        m_enTcpStatus = en_tcp_sts_connected;

        //Start read data...
        if ( Read(m_RecvBuffer, sizeof(m_RecvBuffer)) < 0 )
        {
            ERROR_LOG( "Session(%p), start read failed!", this );
            ret = -4;
            break;
        }

    }while(0);

    DEBUG_LOG( 
        "Session(%p), LocalAddr(%s), RemoteAddr(%s), ret=%d.", 
        this, 
        GetLocalAddr().c_str(), 
        GetRemoteAddr().c_str(), 
        ret 
        );

    return ret;
}

int CTCPHandler::Close()
{
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
        if ( (m_enTcpStatus != en_tcp_sts_connecting )
          && (m_enTcpStatus != en_tcp_sts_connected) )
        {
            WARN_LOG( "Session(%p) is not connected, tcp_status=%d!", this, m_enTcpStatus);
            return 0;
        }

        m_enTcpStatus = en_tcp_sts_closing;

        if( m_Sock.is_open() )
        {
            WARN_LOG( "Session(%p), socket close!", this);

            boost::system::error_code ignored_ec;
            m_Sock.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ignored_ec);
            m_Sock.close();
        }
    }

    OnClosed();
    {
        boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
        m_enTcpStatus = en_tcp_sts_closed;
    }

    DEBUG_LOG( "Session(%p).", this);

    return 0;
}

void CTCPHandler::Reset()
{
    Close();

    m_strRemoteIP.clear();
    m_usRemotePort = 0;

    m_strLocalIP.clear();
    m_usLocalPort = 0;
    m_unLastRwTime = 0;
}

int CTCPHandler::Connect(uint32 ip, uint16 port)
{
    int ret = 0;
    string strIp = inet_ntoa(*(in_addr*)&ip);
    ip::tcp::endpoint remote_ep(ip::address::from_string(strIp), port);

    {
        boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
        if( m_Sock.is_open() && (m_enTcpStatus == en_tcp_sts_connected) )
        {
            WARN_LOG("Session(%p), is already connected", this);
            return 0;
        }
        m_enTcpStatus == en_tcp_sts_connecting;

        switch(m_enTcpMode)
        {
        case en_tcp_on_sync:
            {
                boost::system::error_code ec;
                m_Sock.connect(remote_ep, ec);
                if(!ec)
                {
                    ret = HandleConnect(ec);;
                }
                else
                {
                    ret = -2;
                }
            }
            break;
        case en_tcp_on_async:
            {
                m_Sock.async_connect(
                    remote_ep,
                    boost::bind(&CTCPHandler::HandleConnect, shared_from_this(), boost::asio::placeholders::error ) );
            }
            break;
        case en_tcp_on_aysnc_serial:
            {
                m_Sock.async_connect(
                    remote_ep,
                    make_custom_alloc_handler(
                    m_StrandAllocator,
                    m_Strand.wrap(
                    make_custom_alloc_handler(
                    m_ConnectAllocator,
                    boost::bind( &CTCPHandler::HandleConnect, shared_from_this(), boost::asio::placeholders::error ) ) ) ) );
            }
            break;
        default:
            {
                ret = -3;
            }
            break;
        }
    }
    
    DEBUG_LOG("Session(%p), connect to(%s:%d), ret=%d", this, strIp.c_str(), port, ret);

    return ret;
}

int CTCPHandler::SendMsg(const SDataBuff& data_buff )
{
    /*
    DEBUG_LOG(
        "Session(%p), data_buff_info(%p,%u,%u)", 
        this, 
        data_buff.pbuff_.get(), 
        data_buff.data_size_, 
        data_buff.sent_size_
        );*/

    boost::lock_guard<boost::recursive_mutex> lock(m_SendQueueLock);
    m_SendQueue.push(data_buff);
    if(!m_bSending)
    {
        m_bSending = true;
        SDataBuff send_data = m_SendQueue.front();
        if ( Write(send_data.get_buffer(), send_data.data_size() ) < 0 )
        {
            ERROR_LOG( "Session(%p), send failed, send queue size(%u).", this, m_SendQueue.size() );
            return -1;
        }
    }
    if(m_SendQueue.size()>5)
    {
        WARN_LOG( "Session(%p), send msg(%p,%d), send queue size(%u), send_flag(%d).", 
            this, data_buff.get_buffer(), data_buff.data_size(), m_SendQueue.size(),m_bSending );
    }
    else
    {
        TRACE_LOG( "Session(%p), send msg(%p,%d), send queue size(%u), send_flag(%d).", 
            this, data_buff.get_buffer(), data_buff.data_size(), m_SendQueue.size(),m_bSending );
    }    

    return data_buff.data_size();
}

int CTCPHandler::SendMsg( const void* data, size_t data_len )
{
    SDataBuff data_buff;
    if( !data_buff.copy_data(data, data_len) )
    {
        ERROR_LOG( "Session(%p), copy failed, send queue size(%u).", this, m_SendQueue.size() );
        return -1;
    }

    return SendMsg(data_buff);
}

int CTCPHandler::Write(const void* data_buff, size_t data_len)
{
    int ret = -999;
    if( !data_buff || !data_len)
    {
        return -1;
    }

    std::vector<boost::asio::const_buffer> buffers;
    buffers.push_back(boost::asio::buffer(data_buff, data_len));

    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( !m_Sock.is_open() 
        || (m_enTcpStatus != en_tcp_sts_connected) )
    {
        return -2;
    }

    switch(m_enTcpMode)
    {
    case en_tcp_on_sync:
        {
            boost::system::error_code ec;
            size_t sent_size = boost::asio::write(m_Sock, boost::asio::buffer(data_buff, data_len), ec);
            if(ec)
            {
                ERROR_LOG("Session(%p), Write(%p:%u) error(%s)", this, data_buff, data_len, ec.message().c_str());
                return -3;
            }
            return sent_size;
        }
    case en_tcp_on_async:
        {
#if 0
            boost::asio::async_write(
                m_Sock,
                boost::asio::buffer(data_buff, data_len),
                make_custom_alloc_handler( 
                m_WriteAllocator,
                boost::bind(
                &CTCPHandler::HandleWrite, 
                shared_from_this(), 
                boost::asio::placeholders::error,  
                boost::asio::placeholders::bytes_transferred) ) );
#else
            boost::asio::async_write(m_Sock,
                boost::asio::buffer(data_buff, data_len),
                boost::bind(  
                &CTCPHandler::HandleWrite,  
                shared_from_this(),  
                boost::asio::placeholders::error,  
                boost::asio::placeholders::bytes_transferred) );
#endif
            return 0;
        }
        break;
    case en_tcp_on_aysnc_serial:
        {
            boost::asio::async_write(
                m_Sock,
                boost::asio::buffer(data_buff, data_len),
                make_custom_alloc_handler(
                m_StrandAllocator,
                m_Strand.wrap(
                make_custom_alloc_handler(
                m_WriteAllocator, 
                boost::bind( 
                &CTCPHandler::HandleWrite, 
                shared_from_this(), 
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred ) ) ) ) );
            return 0;
        }
        break;
    default:
        {
            ret = -4;
        }
        break;
    }

    TRACE_LOG("Session(%p), ret=%d", this, ret);

    return ret;
}

int CTCPHandler::Read(void* buff, size_t buff_size)
{
    int ret = -1;
    if( !buff || !buff_size)
    {
        return -1;
    }

    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( !m_Sock.is_open() 
        || (m_enTcpStatus != en_tcp_sts_connected) )
    {
        return -2;
    }

    switch(m_enTcpMode)
    {
    case en_tcp_on_sync:
        {
            boost::system::error_code ec;
            size_t recv_len = m_Sock.read_some(boost::asio::buffer(buff, buff_size), ec);
            if(ec)
            {
                return -3;
            }
            return recv_len;
        }
    case en_tcp_on_async:
        {
#if 0
            m_Sock.async_read_some(
                boost::asio::buffer(buff, buff_size),
                make_custom_alloc_handler( 
                m_ReadAllocator,
                boost::bind(
                &CTCPHandler::HandleRead, 
                shared_from_this(), 
                boost::asio::placeholders::error, 
                boost::asio::placeholders::bytes_transferred ) ) );
#else
            m_Sock.async_read_some(
                boost::asio::buffer(buff, buff_size),
                boost::bind(
                &CTCPHandler::HandleRead, 
                shared_from_this(), 
                boost::asio::placeholders::error, 
                boost::asio::placeholders::bytes_transferred ) );
#endif
            return 0;
        }
        break;
    case en_tcp_on_aysnc_serial:
        {
            m_Sock.async_read_some(
                boost::asio::buffer(buff, buff_size),
                make_custom_alloc_handler(
                m_StrandAllocator, 
                m_Strand.wrap( 
                make_custom_alloc_handler( 
                m_ReadAllocator,
                boost::bind(
                &CTCPHandler::HandleRead, 
                shared_from_this(), 
                boost::asio::placeholders::error, 
                boost::asio::placeholders::bytes_transferred ) ) ) ) );
            return 0;
        }
        break;
    default:
        {
            ret = -4;
        }
        break;
    }

    TRACE_LOG("Session(%p), ret=%d", this, ret);

    return ret;
}

int CTCPHandler::HandleConnect(const boost::system::error_code& error)
{
    TRACE_LOG("Session(%p), error(%s)", this, error.message().c_str());

    int ret = 0;
    do 
    {
        //boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
        if (error)
        {
            ret = -1;
            OnConnectFailed();
            break;
        }

        if ( (ret = Open()) < 0)
        {
            break;
        }

        if( (ret = OnConnected()) < 0 )
        {
            break;
        }
    } while (0);
    
    return ret;
}

int CTCPHandler::HandleRead(const boost::system::error_code& error, size_t bytes_transferred)
{
    int ret = 0;
    TRACE_LOG("Session(%p), bytes_transferred=%u", this, bytes_transferred);
    do 
    {
        if (error)
        {
            ERROR_LOG("Session(%p), error(%s)", this, error.message().c_str());
            
            //if (error != boost::asio::error::operation_aborted)
            {
                //Close();
            }
            ret = -1;
            break;
        }

        {
            boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
            if( !m_Sock.is_open() 
                || (m_enTcpStatus != en_tcp_sts_connected) )
            {
                ret = -2;
                ERROR_LOG( "Session(%p), tcp_status=%d!", this, m_enTcpStatus );
                break;
            }

            m_unLastRwTime = time(NULL);
        }
        
        if(OnRecv(m_RecvBuffer, bytes_transferred)<0)
        {
            ret = -3;
            ERROR_LOG( "Session(%p), handle recv failed!", this );
            break;
        }

        //Start read next data...
        ret = Read(m_RecvBuffer, sizeof(m_RecvBuffer));
        if ( ret < 0 )
        {
            ERROR_LOG( "Session(%p), start next read failed!", this );
            break;
        }

        return 0;

    } while (0);
    
    Close();

    return ret;
}

int CTCPHandler::HandleWrite(const boost::system::error_code& e, size_t bytes_transferred)
{
    TRACE_LOG( "Session(%p), bytes_transferred=%u.", this, bytes_transferred );
    int ret = 0;
    do 
    {
        //检查是否发送异常
        if (e)
        {
            ERROR_LOG("Session(%p)， error(%s)", this, e.message().c_str());
            //if (e != boost::asio::error::operation_aborted)
            {
                //Close();
            }
            ret = -1;
            break;
        }

        //检查tcp状态
        {
            boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
            if( !m_Sock.is_open() 
                || (m_enTcpStatus != en_tcp_sts_connected) )
            {
                ret = -2;
                break;
            }
            m_unLastRwTime = time(NULL);
        }

        //写接口处理
        ret = OnWrite(bytes_transferred);
        if ( ret < 0 )
        {
            break;
        }

        //Send next...
        boost::lock_guard<boost::recursive_mutex> lock(m_SendQueueLock);
        if( !m_SendQueue.empty() )
        {
            TRACE_LOG("Session(%p), write_size(%u), que_size(%u), buff_info(%p,%u).",
                this, 
                bytes_transferred,
                m_SendQueue.size(),
                m_SendQueue.front().pbuff_.get(),
                m_SendQueue.front().data_size_);

            //check whether queue front data is sent complete
            if(bytes_transferred == m_SendQueue.front().data_size_)
            {
                m_SendQueue.pop();
            }
            else if( bytes_transferred < m_SendQueue.front().data_size_)
            {
                m_SendQueue.front().pop_front(bytes_transferred);
                m_SendQueue.front().data_size_ -= bytes_transferred;
            }
            else 
            {
                ERROR_LOG("Session(%p), write error, bytes_transferred(%u), data_size(%u), send queue size(%u).", 
                    this, 
                    bytes_transferred,
                    m_SendQueue.front().data_size_,
                    m_SendQueue.size());
                break;
            }

            if( !m_SendQueue.empty() )
            {
                if( Write(m_SendQueue.front().get_buffer(), m_SendQueue.front().data_size()) < 0 )
                {
                    ERROR_LOG("Session(%p), send next msg failed, send queue size(%u).", this, m_SendQueue.size());
                    return -1;
                }
            }
            else
            {
                m_bSending = false; //clear
            }
            TRACE_LOG( "Session(%p), send queue size(%u).", this, m_SendQueue.size() );
        }

        return 0;
    } while (0);
    
    Close();
    return ret;
}

string CTCPHandler::GetLocalAddr()
{
    string addr_s = m_strLocalIP;
    addr_s += ":";
    addr_s += boost::lexical_cast<string>( m_usLocalPort );
    return addr_s;
}

uint32 CTCPHandler::GetLocalIP()
{
    return inet_addr( m_strLocalIP.c_str() );
}

uint32 CTCPHandler::GetLocalPort()
{
    return (uint32)m_usLocalPort;
}

string CTCPHandler::GetRemoteAddr()
{
    string addr_s = m_strRemoteIP;
    addr_s += ":";
    addr_s += boost::lexical_cast<string>( m_usRemotePort );
    return addr_s;
}

uint32 CTCPHandler::GetRemoteIP()
{
    return inet_addr( m_strRemoteIP.c_str() );
}

uint32 CTCPHandler::GetRemotePort()
{
    return (uint32)m_usRemotePort;
}

uint32 CTCPHandler::GetSendQueueSize()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_SendQueueLock);
    return m_SendQueue.size();
}