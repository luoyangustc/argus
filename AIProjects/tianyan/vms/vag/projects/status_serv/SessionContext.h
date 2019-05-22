#ifndef STATUS_SERVER_SESSIONCONTENXT_H
#define STATUS_SERVER_SESSIONCONTENXT_H

#include <string>
#include <vector>
#include <set>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include "base/include/typedefine.h"
#include "base/include/HostInfo.h"
#include "netlib_framework/include/ITCPSessionSendSink.h"
#include "protocol/include/protocol_status.h"
#include "IStatusMgr.h"

class SessionContext : boost::noncopyable
{
public:
    SessionContext();	
    ~SessionContext();

    bool OnLogin( ITCPSessionSendSink*sink, const CHostInfo& peer_addr, const protocol::StsLoginReq& req, protocol::StsLoginResp& resp );
    bool OnLoadReport( ITCPSessionSendSink*sink, const CHostInfo& peer_addr, const protocol::StsLoadReportReq& req, protocol::StsLoadReportResp& resp );
    bool OnStatusReport( ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds );
    bool OnClose( ITCPSessionSendSink* send_sink, CHostInfo& peer_addr );
private:
    std::string ToIpString( uint32 ip );
    std::string ToString( int int_to_string );
    std::string EpType2String();
public:
    boost::recursive_mutex lock_;
    ITCPSessionSendSink* send_sink_;
    CHostInfo peer_addr_;

    protocol::EndPointType ep_type_;
    uint16 http_port_;
    uint16 serv_port_;
    uint8 num_listen_ip_;
    std::vector<std::string> listen_ip_list_;

    uint16 tcp_conn_num_;
    uint16 cpu_use_;
    uint16 mempry_use_;

    IStatusMgrPtr status_mgr_;
};

typedef boost::shared_ptr<SessionContext> SessionContextPtr;

#endif  // STATUS_SERVER_SESSIONCONTENXT_H
