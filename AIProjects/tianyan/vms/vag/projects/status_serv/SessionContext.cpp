#include "SessionContext.h"
#include <arpa/inet.h>
#include "base/include/GetTickCount.h"
#include "base/include/tick.h"
#include "base/include/variant.h"
#include "base/include/logging_posix.h"
#include "StreamMgr.h"
#include "DeviceMgr.h"
#include "StatusServer.h"

SessionContext::SessionContext()
    : send_sink_(NULL)
    , peer_addr_()
    , ep_type_(protocol::EP_UNKNOWN)
    , http_port_(0)
    , serv_port_(0)
    , num_listen_ip_(0)
    , tcp_conn_num_(0)
    , cpu_use_(0)
    , mempry_use_(0)
    , status_mgr_()
{
}

SessionContext::~SessionContext()
{
}

bool SessionContext::OnLogin( ITCPSessionSendSink* sink, const CHostInfo& peer_addr,const protocol::StsLoginReq& req, protocol::StsLoginResp& resp )
{
    if( req.mask&0x01 == 0 )
    {
        return false;
    }

    send_sink_ = sink;
    peer_addr_ = peer_addr;
  
    ep_type_ = (protocol::EndPointType)(req.ep_type);
    if( ep_type_ == protocol::EP_SMS )
    {
        status_mgr_ = boost::dynamic_pointer_cast<IStatusMgr>( GetDeviceMgr() ); 
    }
    else if( ep_type_ == protocol::EP_STREAM )
    {
        status_mgr_ = boost::dynamic_pointer_cast<IStatusMgr>( GetStreamMgr() ); 
    }
    else
    {
        return false;
    }

    http_port_ = req.http_port;
    serv_port_ = req.serv_port;
    num_listen_ip_ = req.listen_ip_num;
    listen_ip_list_ = req.listen_ips;

    resp.mask = 0x01;
    resp.resp_code = 0;
    resp.load_expected_cycle = 30;  // heartbeat : 30s

    return true;
}

bool SessionContext::OnLoadReport(ITCPSessionSendSink* sink, const CHostInfo& peer_addr, const protocol::StsLoadReportReq& req, protocol::StsLoadReportResp& resp)
{
    if( req.mask & 0x01 )
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        tcp_conn_num_ = req.tcp_conn_num;
        cpu_use_ = req.cpu_use;
        mempry_use_ = req.memory_use;
    }

    resp.mask = 0x01;
    resp.resp_code = 0;
    resp.load_expected_cycle = 20;  // heartbeat : 20s

    return true;
}

bool SessionContext::OnStatusReport(ITCPSessionSendSink* sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds)
{
    if( status_mgr_ )
    {
        return status_mgr_->OnStatusReport(recvds, sendds);
    }

    return false;
}

bool SessionContext::OnClose( ITCPSessionSendSink* send_sink, CHostInfo& peer_addr )
{
    if( status_mgr_ )
    {
        return status_mgr_->OnSessionOffline(peer_addr);
    }
    return true;		
}

std::string SessionContext::EpType2String()
{
    char* ep_type_str = NULL;
    switch (ep_type_)
    {
        case protocol::EP_SMS:
            ep_type_str = (char*)("Session server");
            break;
        case protocol::EP_STREAM:
            ep_type_str = (char*)("Stream server");
            break;
        default:
            ep_type_str = (char*)("Unknow server");
            break;
    }

    return ep_type_str;
}