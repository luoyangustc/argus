#ifndef __STREAM_MGR_H__
#define __STREAM_MGR_H__

#include "IStatusMgr.h"
#include <map>
#include <set>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/shared_ptr.hpp>
#include "base/include/typedefine.h"
#include "base/include/HostInfo.h"
#include "base/include/datastream.h"
#include "protocol/include/protocol_status.h"
#include "Stream.h"

using namespace std;

class StreamMgr : public IStatusMgr
{
public:
    StreamMgr();
    virtual ~StreamMgr();
    void Update();
    virtual int OnStatusReport(CDataStream& recvds, CDataStream& sendds);
    virtual int OnSessionOffline(const CHostInfo& hi_remote);
private:
    bool HandleStreamStatus(protocol::StsStreamStatusReportReq& req);
private:
    boost::recursive_mutex lock_;
    map<string, StreamPtr> all_streams_;
    map<CHostInfo, set<string> > session_streams_;
};

typedef boost::shared_ptr<StreamMgr> StreamMgrPtr;

#endif  // __STREAM_MGR_H__
