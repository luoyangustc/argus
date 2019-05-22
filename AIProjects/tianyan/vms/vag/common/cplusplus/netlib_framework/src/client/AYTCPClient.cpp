
#include "AYTCPClient.h"

#ifdef _WINDOWS
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif
#include <boost/shared_ptr.hpp>
#include "datastream.h"
#include "Log.h"


CAYTCPClient::CAYTCPClient(boost::asio::io_service& io_serv)
    :CTCPClient(io_serv)
{
    m_Reader.Reset();
}

CAYTCPClient::~CAYTCPClient()
{
}

int CAYTCPClient::OnRecv(const void* data, size_t data_len)
{
    do 
    {
        int ret = m_Reader.Push(data, data_len);
        if(ret<0)
        {
            ERROR_LOG("Client(%p), read failed", this, ret);
            break;
        }
        if(!m_pSink)
        {
            break;
        }

        std::queue<SDataBuff> msg_que;
        m_Reader.PopMsgQueue(msg_que);
        while(msg_que.size())
        {
            int ret = m_pSink->OnTCPMessage(m_unRemoteIp, m_unRemotePort, (uint8*)msg_que.front().get_buffer(), msg_que.front().data_size());
            if( ret < 0 )
            {
                ERROR_LOG( "Session(%p), handle tcp message failed, ret=%d!", this, ret );
                return -1;
            }
            msg_que.pop();
        }

        return 0;
    } while (0);
    return -1;
}
