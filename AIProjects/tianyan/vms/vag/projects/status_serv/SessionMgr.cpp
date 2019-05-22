#include "SessionMgr.h"
#include <boost/thread/lock_guard.hpp>
#include "base/include/logging_posix.h"

SessionMgr::SessionMgr()
{
}

SessionMgr::~SessionMgr()
{
}

void SessionMgr::Update()
{


}

bool SessionMgr::OnLogin(ITCPSessionSendSink* send_sink, const CHostInfo& peer_addr, const protocol::StsLoginReq& req, protocol::StsLoginResp& resp)
{
    
    if( req.ep_type != protocol::EP_SMS &&
        req.ep_type != protocol::EP_STREAM )
    {
        Error("recv login msg incorrect, from(%s),ep_type(%d)\n",
            peer_addr.GetNodeString().c_str(), req.ep_type );
        return false;
    }

    SessionContextPtr pSession( new SessionContext() );
    if( !pSession || !pSession->OnLogin(send_sink, peer_addr, req, resp) )
    {
        Error("memory alloc for SessionContext failed, from(%s),ep_type(%d)\n",
            peer_addr.GetNodeString().c_str(), req.ep_type );
        return false;
    }

    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        session_servs_[peer_addr] = pSession;
    }
    
    Debug( "recv login msg, from(%s),ep_type(%d)\n", peer_addr.GetNodeString().c_str(), req.ep_type );

    return true;
}

bool SessionMgr::OnLoadReport(ITCPSessionSendSink* send_sink, const CHostInfo& peer_addr, const protocol::StsLoadReportReq& req, protocol::StsLoadReportResp& resp)
{
    SessionContextPtr pSession = GetSessionContext(peer_addr);
    if( !pSession )
    {
        return false;
    }

    return pSession->OnLoadReport(send_sink, peer_addr, req, resp);
}

bool SessionMgr::OnStatusReport(ITCPSessionSendSink* send_sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds)
{
    SessionContextPtr pSession = GetSessionContext(peer_addr);
    if( !pSession )
    {
        return false;
    }
    return pSession->OnStatusReport(send_sink, peer_addr, recvds, sendds);
}

bool SessionMgr::OnClose(ITCPSessionSendSink* send_sink, CHostInfo& peer_addr)
{
    SessionContextPtr pSession = GetSessionContext(peer_addr);
    if( !pSession )
    {
        return false;
    }
    
    pSession->OnClose(send_sink, peer_addr);
    
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        session_servs_.erase(peer_addr);
    }
    return true;
}

SessionContextPtr SessionMgr::GetSessionContext(const CHostInfo& peer_addr)
{
    SessionContextPtr pSession;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        SessionContextIterator it = session_servs_.find(peer_addr);
        if( it != session_servs_.end() )
        {
            pSession = it->second;
        }
    }
    return pSession;
}

int SessionMgr::GetSessionContext(protocol::EndPointType ep_type, vector<SessionContextPtr>& session_ctxs)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    std::map<CHostInfo, SessionContextPtr>::iterator it = session_servs_.begin();
    for( ; it!= session_servs_.end(); ++it )
    {
        SessionContextPtr pSession = it->second;
        if( pSession->ep_type_ == ep_type )
        {
            session_ctxs.push_back(pSession);
        }
    }
    return session_ctxs.size();
}

SessionContextPtr SessionMgr::SelectBestSessionContext(const CHostInfo& hi_remote, protocol::EndPointType ep_type)
{
    uint32 min_weight = 0xffffffff;
    SessionContextPtr pBestSession;

    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    std::map<CHostInfo, SessionContextPtr>::iterator it = session_servs_.begin();
    for( ; it!= session_servs_.end(); ++it )
    {
        SessionContextPtr pSession = it->second;
        if( pSession->ep_type_ == ep_type )
        {
            uint32 weight = 0;
            if( pSession->cpu_use_ > 80 )
            {
                weight |= ( (uint8)pSession->cpu_use_ << 24 ) & 0xff000000;
            }

            if( pSession->mempry_use_ > 80 )
            {
                weight |= ( (uint8)pSession->mempry_use_ << 16 ) & 0x00ff0000;
            }

            weight |= pSession->tcp_conn_num_;

            if( weight < min_weight )
            {
                min_weight = weight;
                pBestSession = pSession;
            }

        }
    }

    return pBestSession;
}