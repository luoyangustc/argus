#ifndef __MEDIA_SESSION_BASE_H__
#define __MEDIA_SESSION_BASE_H__

#include "CommonInc.h"
#include "variant.h"

class CMediaSessionBase : public boost::enable_shared_from_this<CMediaSessionBase>
{
public:
    CMediaSessionBase(const string& session_id, EnMediaSessionType session_type, const SDeviceChannel& dc);
    ~CMediaSessionBase();
public:
    bool IsRunning();
    string GetSessionID(){return session_id_;}
    string GetSessionTypeString();
    EnMediaSessionType GetSessionType(){return session_type_;}
    SDeviceChannel GetSessionDC(){return session_dc_;}
public:
    virtual bool Start() = 0;
    virtual bool Stop() = 0;
    virtual void Update() = 0;
    virtual void DumpInfo(Variant& info){}
public:
    virtual bool OnTcpClose(const CHostInfo& hiRemote) = 0;
public:
    virtual bool OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaConnectReq& req){return false;}
    virtual bool OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaDisconnectReq& req){return false;}
    virtual bool OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaStatusReq& req){return false;}
    virtual bool OnPlayReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPlayReq& req){return false;}
    virtual bool OnPauseReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPauseReq& req){return false;}
    virtual bool OnCmdReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaCmdReq& req){return false;}
public:
    virtual bool OnPlayResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayResp& resp){return false;}
    virtual bool OnPauseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseResp& resp){return false;}
    virtual bool OnMediaCmdResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdResp& resp){return false;}
    virtual bool OnCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp){return false;}
public:
    virtual bool OnFrameNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaFrameNotify& notify){return false;}
    virtual bool OnEosNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaEosNotify& notify){return false;}
protected:
    boost::recursive_mutex lock_;
    bool running_;
    string session_id_;
    EnMediaSessionType session_type_;
    SDeviceChannel session_dc_;
};

typedef boost::shared_ptr<CMediaSessionBase> CMediaSessionBase_ptr;

#endif