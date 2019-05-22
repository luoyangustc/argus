#ifndef __WEB_USER_CONTEXT_H__
#define __WEB_USER_CONTEXT_H__

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread.hpp>
#include "base/include/HostInfo.h"
#include "base/include/DeviceChannel.h"
#include "base/include/tick.h"
#include "netlib_framework/include/ITCPSessionSendSink.h"
#include "MediaSession.h"
#include "DeviceContext.h"

using namespace protocol;

class CWebUserContext: public boost::enable_shared_from_this<CWebUserContext>
{
public:
    CWebUserContext( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq );
    ~CWebUserContext();

    bool SendResponse( SHttpResponsePara_ptr pResp );
    bool SendResponse(const string& resp_code, const string& content_type, const uint8* content, size_t content_len);
    bool HandleTimeout();
public:
    virtual bool IsAlive();
    virtual void Update();
    virtual void OnTcpClose(const CHostInfo& hiRemote);
public:
    CHostInfo GetRemote(){return hi_remote_;}
    string GetReqUrl() {return http_req_->header_detail->url_;}
protected:
    boost::recursive_mutex lock_;
    bool running_;

    ITCPSessionSendSink* send_sink_;
    CHostInfo hi_remote_;
    SHttpRequestPara_ptr http_req_;
    uint32_t timeout_ms_;
    tick_t last_active_tick_;
};

class CWebUserContext_Live: public CWebUserContext, public IMediaSessionSink
{
public:
    CWebUserContext_Live( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq );
    ~CWebUserContext_Live();
public:
    virtual void OnMediaOpenAck(int err_code, const SDeviceChannel& dc, SessionType session_type, const string& session_id, const CHostInfo& hi_stream_serv);
public:
    bool SendWebLiveReqToStream(const CHostInfo& hiRemote);
    static int OnWebLiveRespFromStream(uint32_t request_id, void* user_data, int err_code, int http_code, const char* http_resp);
private:
    SDeviceChannel dc_;
    string req_type_;   //rtmp, hls, hflv, close
};

class CWebUserContext_Snap: public CWebUserContext, public IDeviceSnapSink
{
public:
    CWebUserContext_Snap( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq );
    ~CWebUserContext_Snap();
public:
    virtual void OnDeviceSnapAck(int err_code, const string& err_msg, const string& pic_url="");
private:
    string device_id_;
    uint16 channel_id_;
    bool is_preview_;
};

typedef boost::shared_ptr<CWebUserContext> CWebUserContext_ptr;
typedef boost::shared_ptr<CWebUserContext_Live> CWebUserContext_Live_ptr;
typedef boost::shared_ptr<CWebUserContext_Snap> CWebUserContext_Snap_ptr;

#endif //__WEB_USER_CONTEXT_H__
