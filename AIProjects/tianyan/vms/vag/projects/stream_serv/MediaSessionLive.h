#ifndef __MEDIA_SESSION_LIVE_H__
#define __MEDIA_SESSION_LIVE_H__

#include "CommonInc.h"
#include "CloudStorageAgent.h"
#include "RtmpLive.h"
#include "MediaSessionBase.h"
#include "PuStream.h"
#include "CuStream.h"
#include "variant.h"

class CMediaSessionLive:public CMediaSessionBase
{
public:
    CMediaSessionLive(const string& session_id, EnMediaSessionType session_type, const SDeviceChannel& dc);
    ~CMediaSessionLive();
    virtual void Update();
    virtual bool Start();
    virtual bool Stop();
    virtual bool OnTcpClose(const CHostInfo& hiRemote);
    virtual void DumpInfo(Variant& info);
public:
    int SetCloudStorage(bool onoff);
    int SetRtmpLiveAgent(bool onoff);
public: //request msg
    virtual bool OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaConnectReq& req);
    virtual bool OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaDisconnectReq& req);
    virtual bool OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaStatusReq& req);
    virtual bool OnPlayReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPlayReq& req);
    virtual bool OnPauseReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPauseReq& req);
    virtual bool OnCmdReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaCmdReq& req);
public: //response msg
    virtual bool OnPlayResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayResp& resp);
    virtual bool OnPauseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseResp& resp);
    virtual bool OnMediaCmdResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdResp& resp);
    virtual bool OnCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp);
public: //notify msg
    virtual bool OnFrameNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaFrameNotify& notify);
private:
    bool OnDeviceConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp);
    bool OnUserConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp);
    bool OnDeviceDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp);
    bool OnUserDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp);
    bool OnDeviceStatus(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp);
    bool OnUserStatus(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp);
    bool OnDeviceCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp);
	bool OnUserCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp);
	void OnStreamStatusReport(int iStatusType, const CHostInfo& hiRemote);
private:
	bool OnStream(MI_FrameData_ptr frame_data);
	void HandlePuUpdate();
	void HandleCuUpdate();
    bool CanClose();
private:
    bool IsAudioCloseEnable();
    bool IsVideoCloseEnable();
    bool IsPlayEnable();
    int GetCuStreamNum();
    CCuStream_ptr GetCuStream(const CHostInfo& hiRemote);
    void AddCuStream(CCuStream_ptr pStream);
	void RemoveCuStream(const CHostInfo& hiRemote);
    EndPointType GetEndpointType(const CHostInfo& hiRemote);
private:
    CPuStream_ptr pu_stream_;
    map<CHostInfo, CCuStream_ptr> hi_cu_streams_;
    map<string, set<CCuStream_ptr> > cu_streams_;  //user_name->cu_steams
    CCloudStorageAgent_ptr cloud_storage_agent_;
    CRtmpLive_ptr rtmp_live_agent_;
private:
    tick_t start_tick_;
    tick_t last_active_tick_;
    tick_t wait_cu_conn_tick_;
    tick_t wait_pu_conn_tick_;
};

typedef boost::shared_ptr<CMediaSessionLive> CMediaSessionLive_ptr;

#endif
