#ifndef __USER_CONTEXT_H__
#define __USER_CONTEXT_H__

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include "base/include/HostInfo.h"
#include "base/include/DeviceChannel.h"
#include "base/include/tick.h"
#include "protocol/include/protocol_client.h"
#include "protocol/include/protocol_device.h"
#include "netlib_framework/include/ITCPSessionSendSink.h"
#include "MediaSession.h"

using namespace protocol;
using namespace std;

class CUserContext : public IMediaSessionSink, public boost::enable_shared_from_this<CUserContext>
{
public:
    CUserContext();
	~CUserContext();
public:
    bool IsAlive();
    void Update();
    bool SendMessage(const char* data_buff, uint32 data_size);
	void OnTcpClose(const CHostInfo& hiRemote);
public:
    virtual void OnMediaOpenAck( int err_code, const SDeviceChannel& dc, SessionType session_type, const string& session_id, const CHostInfo& hi_stream_serv );
public:
    bool ON_CuLoginRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuLoginReq& req,  CuLoginResp& resp );
    bool ON_CuStatusReport( ITCPSessionSendSink* sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuStatusReportReq& req, CuStatusReportResp& resp );
    bool ON_CuMediaOpenRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaOpenReq& req );
    bool ON_CuMediaCloseRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaCloseReq& req,  CuMediaCloseResp& resp );
public:
    SDeviceChannel GetDC(){ return dc_; }
    CHostInfo GetRemote(){ return hi_remote_; }
    string GetUserName(){ return user_name_; }
    int GetSessionType() { return (int)session_type_; }
private:
    boost::recursive_mutex lock_;
    bool running_;
    ITCPSessionSendSink* send_sink_;
    string user_name_;
    SDeviceChannel dc_;     // for client request
    CHostInfo hi_remote_;
    CHostInfo hi_private_;
	token_t token_;

    tick_t last_active_tick_;

    uint32 open_media_seq_;
    tick_t open_media_tick_;
    int session_type_;
};

typedef boost::shared_ptr<CUserContext> CUserContext_ptr;

#endif //__C3_USER_CONTEXT_H__
