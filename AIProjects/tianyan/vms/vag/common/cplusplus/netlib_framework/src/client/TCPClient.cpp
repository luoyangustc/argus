
#include "TCPClient.h"

#ifdef _WINDOWS
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif
#include <boost/shared_ptr.hpp>
#include "Log.h"

CTCPClient::CTCPClient(boost::asio::io_service& io_serv)
    :CTCPHandler(io_serv)
    ,m_unRemoteIp(0)
    ,m_unRemotePort(0)
{
}

CTCPClient::~CTCPClient()
{
    m_pSink = NULL;
}

int CTCPClient::AdviseSink(ITCPClientSink* sink)
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( !sink )
    {
        return -1;
    }

    m_pSink = sink;

    return 0;
}

int CTCPClient::UnadviseSink()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    m_pSink = NULL;
    return 0;
}

int CTCPClient::Connect(uint32 ip, uint16 port)
{
    m_unRemoteIp = ip;
    m_unRemotePort = port;
    return CTCPHandler::Connect(ip, port);
}

int CTCPClient::Send(const uint8* data, uint32 data_len)
{
    do
    {
        if(!CanSend())
        {
            break;
        }

        return SendMsg(data, data_len);

    } while (0);  

    return -1;
}

int CTCPClient::Close()
{   
    return CTCPHandler::Close();
}

bool CTCPClient::CanSend()
{
    do 
    {
        if(!IsOpen())
        {
            WARN_LOG( "Client(%p), handle not open!", this );
            break;
        }

        uint32 send_queue_size = GetSendQueueSize();
        if ( send_queue_size > 100 )
        {
            WARN_LOG( "Client(%p), send queue is overflow, %u!", this, send_queue_size );
            break;
        }       

        return true;

    } while (0);
 
    return false;
}

int CTCPClient::OnConnected()
{
    do 
    {
        if(!m_pSink)
        {
            break;
        }

        return m_pSink->OnTCPConnected(GetRemoteIP(), GetRemotePort());
    } while (0);
    return -1;
}

int CTCPClient::OnConnectFailed()
{
    do 
    {
        if(!m_pSink)
        {
            break;
        }

        return m_pSink->OnTCPConnectFailed(m_unRemoteIp, m_unRemotePort);
    } while (0);
    return -1;
}

int CTCPClient::OnRecv(const void* data, size_t data_len)
{
    do 
    {
        if(!m_pSink)
        {
            break;
        }

        return m_pSink->OnTCPMessage( GetRemoteIP(), GetRemotePort(), (unsigned char*)data, data_len );
    } while (0);
    return -1;
}

int CTCPClient::OnWrite( size_t data_len )
{
    return 0;
}

int CTCPClient::OnClosed()
{
    do 
    {
        if(!m_pSink)
        {
            break;
        }

        return m_pSink->OnTCPClose( m_unRemoteIp, m_unRemotePort );
    } while (0);
    return -1;
}
