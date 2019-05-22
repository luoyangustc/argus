#include "StatusServer.h"
#include <sys/time.h>
#include "base/include/tick.h"
#include "base/include/http_header_util.h"
#include "base/include/logging_posix.h"
#include "protocol/include/protocol_status.h"
#include "DeviceMgr.h"
#include "StreamMgr.h"
//#include "RedisAccess.h"

StatusServer* GetService()
{
    return StatusServer::ServerInstance();
}

DeviceMgrPtr GetDeviceMgr()
{
    return GetService()->SessionDevMgrInstance();
}

StreamMgrPtr GetStreamMgr()
{
    return GetService()->StreamDevMgrInstance();
}

std::string TimestampStr(const protocol::STimeVal& tv)
{
    char szbuffer[64] = "";
    char* position = szbuffer;
    size_t len = strftime(position, sizeof szbuffer, "%Y-%m-%d %H:%M:%S", localtime((time_t*)&tv.tv_sec));
    if ( 0 == len || snprintf(position + len, sizeof(szbuffer) - len, ".%llu", tv.tv_usec) < 0)
    {
        return std::string();
    }
    else
    {
        return szbuffer;
    }
}

IServerLogical* IServerLogical::GetInstance()
{
    return StatusServer::ServerInstance();
}

StatusServer::StatusServer()
    : session_mgr_(),
    http_request_handler_(),
    http_port_(9010),
    serv_port_(9000)
{
}

StatusServer::~StatusServer()
{
}

bool StatusServer::Start(uint16 http_port,uint16 server_port)
{
    http_port_ = http_port;
    serv_port_ = server_port;

    session_mgr_.reset(new SessionMgr);
    if(!session_mgr_)
    {
        return false;
    }

    device_mgr_.reset(new DeviceMgr());
    if(!device_mgr_)
    {
        return false;
    }

    stream_mgr_.reset(new StreamMgr());
    if(!stream_mgr_)
    {
        return false;
    }

    http_request_handler_.reset(new HttpRequestHandler(session_mgr_));
    if(!http_request_handler_)
    {
        return false;
    }

    //return RedisAccessSingleton::instance().Initialize();
    return true;
}

void StatusServer::Stop()
{
}

int32 StatusServer::OnTCPMessage(ITCPSessionSendSink*sink, CHostInfo& peer_addr, uint32 msg_id, CDataStream& recvds, CDataStream& sendds)
{
    Trace("msgid(%x),from(%s),size(%u)",
        msg_id,
        peer_addr.GetNodeString().c_str(),
        recvds.getbuffer_length());

    switch (msg_id)
    {
    case protocol::MSG_ID_STS_LOGIN:
        {
            if (OnLogin(sink,peer_addr,recvds,sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_STS_LOAD_REPORT:
        {
            if (OnLoadReport(sink,peer_addr,recvds,sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_STS_SESSION_STATUS_REPORT:
    case protocol::MSG_ID_STS_STREAM_STATUS_REPORT:
        {
            if (OnStatusReport(sink,peer_addr,recvds,sendds))
            {
                return 0;
            }
        }
        break;
    default:
        {
            Warn("status server not support message(%x) from(%s:%u)!",
                msg_id,
                peer_addr.GetNodeString().c_str(),
                peer_addr.Port);
        }
        break;
    };	

    return -1;
}

int32 StatusServer::OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& peer_addr, CDataStream& sendds)
{
    (void)sink;  (void)sendds;
    Info("accepted from(%s) !", peer_addr.GetNodeString().c_str());
    return  0;
}

int32 StatusServer::OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& peer_addr)
{
    Debug("closed from(%s)!", peer_addr.GetNodeString().c_str());

    if (session_mgr_)
    {		        
        session_mgr_->OnClose(sink, peer_addr);
    }

    return 0;  
}

void StatusServer::Update()
{
    if (session_mgr_)
    {
        session_mgr_->Update();
    }

    // HeartBeat is at least 30 seconds for between redis client and server
    //const int heartbeat_cycle = 30*1000;  // 30s
    //RedisAccessSingleton::instance().Keepalive(heartbeat_cycle);
}

void StatusServer::DoIdleTask()
{
    //  if (session_mgr_)
    //  {
    //    session_mgr_->DoIdleTask();
    //  }
}

int32 StatusServer::OnHttpClientRequest(ITCPSessionSendSink*sink,CHostInfo& peer_addr,
    SHttpRequestPara_ptr http_request,
    SHttpResponsePara_ptr http_resp)
{
    (void)sink;

    BOOST_ASSERT(http_request);
    BOOST_ASSERT(http_resp);

    do 
    {
        if (! http_request->header_detail->is_request) break;    

        Debug("from(%s), http request url(%s)",peer_addr.GetNodeString().c_str(),
            http_request->header_detail->url_.c_str());

        http_request_handler_->OnHttpRequest(peer_addr, http_request, http_resp);     

        return http_request->header_data_len;  // return http request length
    } while(false);

    return -1;
}

StatusServer* StatusServer::ServerInstance()
{
    static StatusServer server_instance;
    return &server_instance;
}

bool StatusServer::OnLogin(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds)
{
    Debug("from(%s),recv StsLoginReq message\n", peer_addr.GetNodeString().c_str());

    do 
    {
        protocol::MsgHeader header;
        recvds >> header;
        if (!recvds.good_bit())
        {
            Error("Parse header error\n");
            break;
        }
        protocol::StsLoginReq request;
        recvds >> request;
        if (!recvds.good_bit())
        {
            Error("Parse StsLoginReq Message Error!\n");
            break;
        }

        // FIXME : who should respond to client's request ?
        protocol::StsLoginResp resp;
        if ( session_mgr_&& session_mgr_->OnLogin(sink,peer_addr,request,resp) )
        {
            header.msg_id = protocol::MSG_ID_STS_LOGIN;
            header.msg_type = protocol::MSG_TYPE_RESP;

            sendds << header;
            sendds << resp;
            *((WORD*)sendds.getbuffer()) = sendds.size();  // header.msg_size

            Debug("post StsLoginResp message to (%s)!\n", peer_addr.GetNodeString().c_str());
            return true;
        }
    } while(false);

    return false;
}

bool StatusServer::OnLoadReport(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds)
{
    Trace("from(%s),recv StsLoadReportReq message\n", peer_addr.GetNodeString().c_str());

    do
    {
        protocol::MsgHeader header;
        recvds >> header;
        if (!recvds.good_bit())
        {
            Error("Parse header error\n");
            break;
        }	
        protocol::StsLoadReportReq request;
        recvds >> request;
        if (!recvds.good_bit())
        {
            Error("from(%s),Parse StsLoadReportReq Message Error!\n", peer_addr.GetNodeString().c_str());
            break;
        }

        // FIXME : who should response client's request ?
        protocol::StsLoadReportResp resp;
        if (session_mgr_&&
            session_mgr_->OnLoadReport(sink,peer_addr,request,resp))
        {
            header.msg_id = protocol::MSG_ID_STS_LOAD_REPORT;
            header.msg_type = protocol::MSG_TYPE_RESP;			

            sendds << header;
            sendds << resp;
            *((WORD*)sendds.getbuffer()) = sendds.size();  //header.msg_size

            Trace("Post StsLoadReportResp message to (%s) \n", peer_addr.GetNodeString().c_str());

            return true;
        }
    }while(false);

    return false;
}

bool StatusServer::OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds)
{
    Trace("Recv StsStatusReportReq from (%s) \n", peer_addr.GetNodeString().c_str());

    // FIXME : who should response client's request ?
    if (session_mgr_->OnStatusReport(sink,peer_addr,recvds, sendds) < 0)
    {
        Error("session_mgr OnStatusReport error \n");
        return false;
    }
    else
    {
        Trace("Post StsStatusReportResp message to (%s) \n", peer_addr.GetNodeString().c_str());
        return true;
    }
}