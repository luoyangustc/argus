#ifndef STATUS_SERVER_SESSIONMGR_H
#define STATUS_SERVER_SESSIONMGR_H

#include <string>
#include <map>
#include <vector>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "base/include/HostInfo.h"
#include "protocol/include/protocol_status.h"
#include "netlib_framework/include/ITCPSessionSendSink.h"
#include "SessionContext.h"

using namespace std;

class SessionMgr : boost::noncopyable
{
public:
    typedef std::map<CHostInfo, SessionContextPtr> SessionContextMap;
    typedef std::map<CHostInfo, SessionContextPtr>::iterator SessionContextIterator;
public: 
    SessionMgr();
    ~SessionMgr();
    void Update();
public:
    bool OnLogin(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, const protocol::StsLoginReq& req, protocol::StsLoginResp& resp);
    bool OnLoadReport(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, const protocol::StsLoadReportReq& req, protocol::StsLoadReportResp& resp);
    bool OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds);
    bool OnClose(ITCPSessionSendSink* send_sink, CHostInfo& peer_addr);
public:
    int GetSessionContext(protocol::EndPointType ep_type, vector<SessionContextPtr>& session_ctxs);
    SessionContextPtr GetSessionContext(const CHostInfo& peer_addr);
    SessionContextPtr SelectBestSessionContext(const CHostInfo& hi_remote, protocol::EndPointType ep_type);
public:
    boost::recursive_mutex lock_;
    std::map<CHostInfo, SessionContextPtr> session_servs_;
};

typedef boost::shared_ptr<SessionMgr> SessionMgrPtr;

#endif  // STATUS_SERVER_SESSIONMGR_H
