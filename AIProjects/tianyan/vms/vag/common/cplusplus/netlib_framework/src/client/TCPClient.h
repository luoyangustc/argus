#ifndef __TCPCLIENTSOCKET_H__
#define __TCPCLIENTSOCKET_H__

#include "Common.h"
#include "IClientSocket.h"
#include "IClientSocketSink.h"
#include "BufferInfo.h"
#include "TCPHandler.h"

using namespace boost::asio;

class CTCPClient: public CTCPHandler, public ITCPClient
{
public:
    CTCPClient(boost::asio::io_service& io_serv);
    virtual ~CTCPClient();
public:
	virtual int AdviseSink(ITCPClientSink* sink);
	virtual int UnadviseSink();
	virtual int Connect(uint32 ip, uint16 port);
	virtual int Send(const uint8* data, uint32 data_len);
	virtual int Close();
	virtual bool CanSend();
private:
    virtual int OnConnected();
    virtual int OnConnectFailed();
    virtual int OnRecv(const void* data, size_t data_len);
    virtual int OnWrite( size_t data_len );
    virtual int OnClosed();
protected:
    boost::recursive_mutex m_Lock;
    ITCPClientSink* m_pSink;
    uint32 m_unRemoteIp;
    uint16 m_unRemotePort;
};

typedef boost::shared_ptr<CTCPClient> CTCPClient_ptr;

#endif //__TCPCLIENTSOCKET_H__

