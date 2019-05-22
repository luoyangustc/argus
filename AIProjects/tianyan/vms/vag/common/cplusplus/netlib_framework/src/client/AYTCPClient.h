#ifndef __AY_TCP_CLIENT_H__
#define __AY_TCP_CLIENT_H__

#include "TCPClient.h"
#include "AYMsgReader.h"
#include "typedefine.h"
#include "BufferInfo.h"
#include "IClientSocket.h"
#include "IClientSocketSink.h"
#include "IoServicePool.h"

using namespace std;

class CAYTCPClient: public CTCPClient
{
public:
    CAYTCPClient(boost::asio::io_service& io_serv);
    virtual ~CAYTCPClient();
private:
    virtual int OnRecv(const void* data, size_t data_len);
private:
    CAYMsgReader m_Reader;
};

#endif
