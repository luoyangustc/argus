//#include "CLog.h"
#include "HandlerAllocator.h"
#include "AYClient.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#define BOOST_DATE_TIME_SOURCE

CAYClient::CAYClient():m_bConnected(false)
{
    m_nClientId = -1;
    m_bConnected = false;
}

CAYClient::~CAYClient()
{
}

int CAYClient::OnConnected(uint32 ip,uint16 port)
{
    m_bConnected = true;
    PrintLog("CAYClient::OnConnected--->%p, end\n", this);
    return 0;
}

int CAYClient::OnConnectFailed(uint32 ip,uint16 port)
{
    m_bConnected = false;
    PrintLog("CAYClient::OnConnectFailed--->%p, end\n", this);
    return 0;
}

int CAYClient::OnClosed(uint32 ip,uint16 port)
{
    m_bConnected = false;
    PrintLog("CAYClient::OnClosed--->%p, end\n", this);
    return 0;
}

//消息类型定义
enum MSG_TYPE
{
    MSG_TYPE_EXCHANGE_REQ   = 0x00000001,   //密钥交换请求
    MSG_TYPE_EXCHANGE_RESP  = 0x00000002,   //密钥交换响应
    MSG_TYPE_CMD_REQ        = 0x00000003,   //Command请求
    MSG_TYPE_CMD_RESP       = 0x00000004,   //Command响应
    MSG_TYPE_ECHO_TEST      = 0x00000005,   //反射测试消息
};

int CAYClient::OnTCPMessage(uint32 ip, uint16 port, uint8* data_buff, uint32 data_len)
{
    CDataStream recvds(data_buff, data_len);
    uint16 msg_size;
    uint32 msg_type;
    recvds >> msg_size;
    recvds >> msg_type;
    //recvds >> header;
            
    PrintLog("Session(%p)--->recv tcp message, msg_size=%u, msg_type=%x.\n", this, msg_size, msg_type);
    switch(msg_type)
    {
    case MSG_TYPE_ECHO_TEST:
        {
            string recv_msg;
            recvds >> recv_msg;

            PrintLog("Session(%p)--->echo:%s\n", this, recv_msg.c_str());
        }
        break;
    default:
        PrintLog( "Session(%p)--->receive from(%x:%d)(0x%x)!\n", this, ip, port, msg_type);
        break;

    }
    return 0;
}

void CAYClient::PrintLog(const char* fmt, ...)
{
    if(m_nClientId%500 != 0)
    {
        return;
    }

    string strTime = get_local_time();
    if(strTime.empty())
    {
        return;
    }

    char new_fmt[1024];
#ifndef _WINDOWS
    int ret = snprintf(new_fmt, sizeof(new_fmt)-1, "%s >>> %s", strTime.c_str(), fmt);
#else
    int ret = _snprintf(new_fmt, sizeof(new_fmt)-1, "%s >>> %s", strTime.c_str(), fmt);
#endif
    if( ret < 0 )
    {
        return;
    }
    new_fmt[ret] = '\0';

    va_list args;
    va_start(args, fmt);
    vprintf(new_fmt, args);
    va_end(args);
}

void CAYClient::SetClientId(int client_id)
{
    m_nClientId = client_id;
    PrintLog( "Session(%p)--->Set client_id(%06d)\n", this, m_nClientId);
}

bool CAYClient::Start(uint32 remote_ip, uint16 remote_port)
{
    do
    {
        PrintLog( "Session(%p)--->connect to %x:%d...\n", this, remote_ip, remote_port);
        m_spTcpSocket = CClientSocketFactory::CreateTCPClientSocketObj();
        if(!m_spTcpSocket)
        {
            PrintLog( "Session(%p)--->create tcp socket failed, client_id=%d!\n", this, m_nClientId);
            break;
        }
        m_spTcpSocket->AdviseSink(this);
        if( m_spTcpSocket->Connect(remote_ip, remote_port) < 0 )
        {
            PrintLog("Session(%p)--->connect to server failed!\n", this);
            return false;
        }
        return true;
    } while (0);

    return false;
}

bool CAYClient::Stop()
{
    if(m_spTcpSocket)
    {
        m_spTcpSocket->Close();
    }

    PrintLog( "Session(%p)--->client(%06d) Stop!\n", this, m_nClientId);

    return true;
}

bool CAYClient::Heartbeat(uint32 cnt)
{
    char msg[64];
    int len = sprintf(msg, "client(%06d) heartbeat(%u)", m_nClientId, cnt);
    if(len < 0)
    {
        return false;
    }
    msg[len++] = '\0';

    char send_buff[1024];
    CDataStream sendds(send_buff, sizeof(send_buff));
    uint16 msg_size = sizeof(uint16) + sizeof(uint32) + len;
    uint32 msg_type = MSG_TYPE_ECHO_TEST;
    sendds << msg_size;
    sendds << msg_type;
    sendds.writedata(msg, len);

    if( !SendData(sendds.getbuffer(), sendds.size()) )
    {
        boost::this_thread::sleep( boost::posix_time::millisec(100));
        PrintLog( "Session(%p)--->client(%06d) send heartbeat msg failed!\n", this, m_nClientId);
        return false;
    }
    
    ++m_nHeartbeatCnt;

    PrintLog( "Session(%p)--->%s.\n", this, msg);

    return true;
}

bool CAYClient::SendData( const void* data_buff, size_t data_len )
{
    if(!m_spTcpSocket || !m_spTcpSocket->CanSend())
    {
        return false;
    }

    if( m_spTcpSocket->Send((uint8*)data_buff, data_len) < 0 )
    {
        return false;
    }

    return true;
}

string CAYClient::get_local_time()
{
    char   strTime[128];
    memset(strTime, 0x0, sizeof(strTime));

#ifndef _WINDOWS

    struct timeval cur_tv;
    struct timezone tz;
    gettimeofday(&cur_tv,&tz);
    struct tm* local_time = localtime((time_t*)&cur_tv.tv_sec);
    sprintf(strTime, "%d-%d-%d %d:%d:%d:%d", 
        local_time->tm_year + 1900, 
        local_time->tm_mon+1, 
        local_time->tm_mday,
        local_time->tm_hour,
        local_time->tm_min, 
        local_time->tm_sec, 
        (int)cur_tv.tv_usec/1000 );
#else    
    SYSTEMTIME sys;
    GetLocalTime( &sys );
    sprintf(strTime, "%d-%d-%d %d:%d:%d:%d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour,sys.wMinute,sys.wSecond, sys.wMilliseconds);
#endif
    return strTime;
}