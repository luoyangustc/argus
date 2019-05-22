#ifndef __AY_TCP_SESSION_H__
#define __AY_TCP_SESSION_H__

#include "TCPServer.h"
#include <queue>
#include "BufferInfo.h"
#include "ITCPSessionSendSink.h"
#include "../comm/TCPHandler.h"
#include "../comm/AYMsgReader.h"
#include "../exchang_key/AYExchangeKey.h"
#include "typedefine.h"

using namespace std;

#define  MAX_AY_MSG_SIZE (20*1024+512)

class CAYSession: public CTCPHandler, public ITCPSessionSendSink
{
public:
	CAYSession(boost::asio::io_service& io_service);
	virtual ~CAYSession();
    int Init(CTCPServer<CAYSession>* serv);
    virtual int TaskMain();
    boost::shared_ptr<CAYSession> SharedFromThis();
public:
    virtual int OnRecv(const void* data, size_t data_len);
    virtual int OnWrite(size_t data_len);
    virtual int OnClosed();
public:
    virtual int SendFunc( SDataBuff& data_buff );
    virtual int SendFunc( uint8 * data, uint32 data_len );
    virtual uint32 GetSendQLengthFunc();
    virtual uint32 GetSendSpeed( unsigned int recent_second );
    virtual uint32 GetRecvSpeed( unsigned int recent_second );
    virtual CHostInfo GetLocalHost();
    virtual CHostInfo GetRemoteHost();
    virtual std::ostringstream& DumpInfo( std::ostringstream& oss );
private:
    //int OpenTask();
    int CloseTask();
    int RecvQueueTask();
    int HandleExchangeKey(IN CDataStream& recvds, OUT CDataStream& sendds);
private:
    boost::recursive_mutex m_Lock;
    volatile bool m_bInTasking;
    CTCPServer<CAYSession>* m_pServer;
    CAYExchangeKeyServer m_ExchgKey;
    CAYMsgReader m_Reader;
};

#endif
