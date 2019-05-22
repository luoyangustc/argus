#ifndef __MEDIA_SESSION_MGR_H__
#define __MEDIA_SESSION_MGR_H__

#include "CommonInc.h"
#include "DBWriter.h"
#include "MediaSessionBase.h"
#include "MediaSessionLive.h"

class CMediaSessionMgr
{
public:
    CMediaSessionMgr();
    ~CMediaSessionMgr();
    void Update();
    bool Start();
    bool Stop();
    void DumpInfo(Variant& info);
    void DumpInfo(const string& device_id, Variant& info);
    void DumpInfo(const SDeviceChannel& dc, Variant& info);
public:
    CMediaSessionBase_ptr GetMediaSession(string session_id);
    CMediaSessionBase_ptr GetMediaSession(EnMediaSessionType session_type, const SDeviceChannel& dc);
    CMediaSessionBase_ptr CreateMediaSession(string session_id, EnMediaSessionType session_type, const SDeviceChannel& dc );
    void RemoveMediaSession(const string& session_id);
    void RemoveMediaSession(EnMediaSessionType session_type, const SDeviceChannel& dc);
public:
    bool OnTCPClosed( const CHostInfo& hiRemote );
public:
    bool OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaConnectReq& req);
    bool OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaDisconnectReq& req);
    bool OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaStatusReq& req);
    bool OnPlayReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPlayReq& req);
    bool OnPauseReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPauseReq& req);
    bool OnCmdReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaCmdReq& req);
public:
    bool OnPlayResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayResp& resp);
    bool OnPauseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseResp& resp);
    bool OnMediaCmdResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdResp& resp);
    bool OnCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp);
public:
    bool OnFrameNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaFrameNotify& notify);
    bool OnEosNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaEosNotify& notify);
private:
    boost::recursive_mutex lock_;
    map<string, CMediaSessionBase_ptr> media_sessions_;
    map<SDeviceChannel, CMediaSessionBase_ptr> live_sessions_;
};

typedef boost::shared_ptr<CMediaSessionMgr> CMediaSessionMgr_ptr;

#endif