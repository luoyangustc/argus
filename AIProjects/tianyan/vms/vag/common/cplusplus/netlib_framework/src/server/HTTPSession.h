#ifndef __HTTP_SESSION_H__
#define __HTTP_SESSION_H__

#include <queue>
#include "TCPServer.h"
#include "typedefine.h"
#include "BufferInfo.h"
#include "IServerLogical.h"
#include "ITCPSessionSendSink.h"

#include "TCPHandler.h"
#include "HTTPParser.h"
#include "HTTPResponse.h"

using namespace std;

#define  MAX_HTTP_REQUEST_MSG_SIZE (20*1024)

class CHTTPSession: public CTCPHandler, public ITCPSessionSendSink
{
public:
	CHTTPSession(boost::asio::io_service& io_service);
	virtual ~CHTTPSession();

    int Init(CTCPServer<CHTTPSession>* serv);
    virtual int TaskMain();
    void Reset();
    boost::shared_ptr<CHTTPSession> SharedFromThis();
public:
    virtual int OnRecv(const void* data, size_t data_len );
    virtual int OnWrite(size_t data_len);
    virtual int OnClosed();
public:
    virtual int SendHttpResp( SHttpResponsePara_ptr pResp );
    virtual int SendFunc( SDataBuff& data_buff );
    virtual int SendFunc( uint8 * data, uint32 data_len );
    virtual uint32 GetSendQLengthFunc();
    virtual uint32 GetSendSpeed( unsigned int recent_second );
    virtual uint32 GetRecvSpeed( unsigned int recent_second );
    virtual CHostInfo GetLocalHost();
    virtual CHostInfo GetRemoteHost();
    virtual std::ostringstream& DumpInfo( std::ostringstream& oss );
private:
    CTCPServer<CHTTPSession>*   m_pServer;
    SHttpRequestPara_ptr        m_spHttpReqPara;
    SHttpResponsePara_ptr       m_spHttpRespPara;
    CHTTPParser                 m_ReqParser;
    CHttpResponse               m_Response;
    SDataBuff                   m_RequestMsg;
};

#endif