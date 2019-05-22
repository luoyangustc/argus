#ifndef __TCP_HANDLER_H__
#define __TCP_HANDLER_H__

#include <stdio.h>
#include <queue>
#include "Common.h"
#include "HandlerAllocator.h"
#include "BufferInfo.h"
#include "typedefine.h"

using namespace std;
using namespace boost::asio;

#define  MAX_TCP_RECV_BUFF_SIZE (32*1024)
#define  MAX_TCP_SEND_BUFF_SIZE (32*1024)

enum EN_TCP_MODE
{
    en_tcp_on_sync,
    en_tcp_on_async,
    en_tcp_on_aysnc_serial,
    en_tcp_on_max
};

enum EN_TCP_STATUS
{
    en_tcp_sts_init = 0,
    en_tcp_sts_connecting,
    en_tcp_sts_connect_failed,
    en_tcp_sts_connected,
    en_tcp_sts_disconnect,
    en_tcp_sts_closing,
    en_tcp_sts_closed,
    en_tcp_sts_max
};

class CTCPHandler: public boost::enable_shared_from_this<CTCPHandler>  
{
public:
	CTCPHandler( boost::asio::io_service& io_service );
	virtual ~CTCPHandler();
    void SetMode(EN_TCP_MODE mode = en_tcp_on_async);
    int Open();
    int Close();
    void Reset();
    bool IsOpen();
    bool IsAlive();
public:
    int Connect(uint32 ip, uint16 port);
    int SendMsg( const SDataBuff& data_buff );
    int SendMsg( const void* data, size_t data_len );
public:
    virtual int OnConnected(){return 0;}
    virtual int OnConnectFailed(){return 0;}
    virtual int OnRecv( const void* data, size_t data_len ) = 0;
    virtual int OnWrite( size_t data_len ) = 0;
    virtual int OnClosed() = 0;
public:
    TCPSocket& Socket(){ return m_Sock; }
    string GetLocalAddr();
    uint32 GetLocalIP();
    uint32 GetLocalPort();
    string GetRemoteAddr();
    uint32 GetRemoteIP();
    uint32 GetRemotePort();
    uint32 GetSendQueueSize();
private:
    int Write( const void* data_buff, size_t data_len );
    int Read( void* buff, size_t buff_size );

    int HandleConnect( const boost::system::error_code& error );
    int HandleRead( const boost::system::error_code& error, size_t bytes_transferred );
    int HandleWrite( const boost::system::error_code& error, size_t bytes_transferred );
private:
    boost::recursive_mutex      m_Lock;
    io_service&	                m_IoService;
    io_service::strand          m_Strand;
    //TCPSocket_ptr				m_spSocket;
    TCPSocket                   m_Sock;
    EN_TCP_MODE                 m_enTcpMode;

    EN_TCP_STATUS               m_enTcpStatus;
    time_t                      m_unLastRwTime;
    volatile bool               m_bRunning;

    std::string                 m_strLocalIP;
    std::string                 m_strRemoteIP;
    uint16                      m_usLocalPort;
    uint16                      m_usRemotePort;
   
    handler_allocator			m_StrandAllocator;
    handler_allocator			m_ConnectAllocator;
    handler_allocator			m_ReadAllocator;
    handler_allocator			m_WriteAllocator;

    //接受缓冲区
    char m_RecvBuffer[MAX_TCP_RECV_BUFF_SIZE];

    //发送消息列表
    bool m_bSending;
    std::queue<SDataBuff> m_SendQueue;
    boost::recursive_mutex m_SendQueueLock;
    
#if 0   //for test
    FILE*                       m_pSendDataFile;
    FILE*                       m_pSendDataTag;
    unsigned int                m_unSendDataNo;
#endif

};

#endif //__TCP_HANDLER_H__