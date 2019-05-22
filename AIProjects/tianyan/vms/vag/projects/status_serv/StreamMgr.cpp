#include <sys/time.h>
#include "StreamMgr.h"
#include <boost/thread/lock_guard.hpp>
#include <boost/algorithm/string.hpp>
#include "base/include/logging_posix.h"

StreamMgr::StreamMgr()
{

}

StreamMgr::~StreamMgr()
{

}

void StreamMgr::Update()
{

}

int StreamMgr::OnStatusReport(CDataStream& recvds, CDataStream& sendds)
{
    protocol::MsgHeader header;
    recvds >> header;
    if ( !recvds.good_bit() )
    {
        Error("Parse MsgHeader Message Error!\n");
        return -1;
    }

    protocol::StsStreamStatusReportReq req;
    recvds >> req;
    if (!recvds.good_bit())
    {
        Error("Parse StsStreamStatusReportReq Message Error!\n");
        return -2;
    }

    protocol::StsSessionStatusReportResp resp;
    resp.mask = 0x01;
    if ( !HandleStreamStatus(req) )
    {
        resp.resp_code = -10001;
    }
    else
    {
        resp.resp_code = protocol::EN_SUCCESS;
    }

    header.msg_id = protocol::MSG_ID_STS_STREAM_STATUS_REPORT;
    header.msg_type = protocol::MSG_TYPE_RESP;
    sendds << header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();          
    return 0;
}

int StreamMgr::OnSessionOffline(const CHostInfo& hi_remote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    map<CHostInfo, set<string> >::iterator it = session_streams_.find(hi_remote);
    if( it!=session_streams_.end() )
    {
        struct timeval cur_tv;
        gettimeofday(&cur_tv, NULL);

        set<string>& streams = it->second;
        set<string>::iterator itSessionId = streams.begin();
        for( ; itSessionId!=streams.end(); ++itSessionId)
        {
            map<string, StreamPtr>::iterator itStream = all_streams_.find(*itSessionId);
            if( itStream != all_streams_.end() )
            {
                itStream->second->status_ = 0xff;
                itStream->second->timestamp_.tv_sec = (uint64)cur_tv.tv_sec;
                itStream->second->timestamp_.tv_usec = (uint64)cur_tv.tv_usec;
            }

        }
        session_streams_.erase(it);
    }
    return 0;
}

bool StreamMgr::HandleStreamStatus(protocol::StsStreamStatusReportReq& req)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    vector<protocol::SDeviceStreamStatus>::iterator it = req.devices.begin();
    for ( /*void*/; it != req.devices.end(); ++it )
    {
        Debug( "session_id(%s),status(%d), timestamp(%llu.%llu), stream_serv(%s:%d).\n",
            it->session_id.c_str(), it->status, it->timestamp.tv_sec, it->timestamp.tv_usec,
            it->stream_serv_addr.ip.c_str(), it->stream_serv_addr.port );

        const protocol::SDeviceStreamStatus& status = *it;
        StreamPtr pStream;
        map<string, StreamPtr>::iterator itStream = all_streams_.find(status.session_id);
        if ( itStream == all_streams_.end() )  // not found
        {
            pStream.reset(new Stream());

            if ( !pStream || !pStream->OnStatusReport(status) )
            {
                return false;
            }
            all_streams_[status.session_id] = pStream;
        }
        else
        {
            pStream = itStream->second;
            if ( !pStream->OnStatusReport(status) )
            {
                return false;
            }
        }

        if( status.status == protocol::SDeviceStreamStatus::enm_dev_media_connected )
        {
            CHostInfo hiStream(status.stream_serv_addr.ip.c_str(), status.stream_serv_addr.port);
            session_streams_[hiStream].insert(status.session_id);
        }
        else
        {
            std::map<CHostInfo, set<string> >::iterator it = session_streams_.find(pStream->stream_serv_addr_);
            if( it!=session_streams_.end() )
            {
                it->second.erase(pStream->session_id_);
            }
        }
    }

    return true;
}
