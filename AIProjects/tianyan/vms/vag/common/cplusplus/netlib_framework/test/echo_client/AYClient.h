#ifndef __AY_CLIENT_H__
#define __AY_CLIENT_H__

#include <queue>
#include "typedefine.h"
#include "datastream.h"
#include "Common.h"
#include "HandlerAllocator.h"
#include "BufferInfo.h"
#include "client/ITCPClientSocket.h"
#include "client/ITCPClientSocketSink.h"
#include "client/ClientSocketFactory.h"

using namespace std;
using namespace boost::asio;

class CAYClient:public ITCPClientSocketSink
{
public:
	CAYClient();
	virtual ~CAYClient();
    bool Start(uint32 remote_ip, uint16 remote_port);
    bool Stop();
    bool Heartbeat(uint32 cnt);
    bool SendData( const void* data_buff, size_t data_len );
    bool IsConnected(){return m_bConnected;}
    void SetClientId(int client_id);
    void PrintLog(const char* fmt, ...);
    static string get_local_time();
public:
    virtual int OnConnected(uint32 ip, uint16 port);
	virtual int OnConnectFailed(uint32 ip, uint16 port);
	virtual int OnClosed(uint32 ip, uint16 port);
    virtual int OnTCPMessage(uint32 ip, uint16 port, uint8* data_buff, uint32 data_len);
private:
    bool m_bConnected;
    int m_nClientId;
    uint32 m_nHeartbeatCnt;
    ITCPClientSocket_ptr m_spTcpSocket;
};

typedef boost::shared_ptr<CAYClient> CAYClient_ptr;

#endif