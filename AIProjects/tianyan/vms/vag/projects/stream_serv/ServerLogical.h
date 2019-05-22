#ifndef __SERVER_LOGICAL_H__
#define __SERVER_LOGICAL_H__

#include "CommonInc.h"
#include "Config.h"
#include "DBWriter.h"
#include "MediaSessionMgr.h"
#include "MonitorThread.h"
#include "StreamStatusReport.h"

class CServerLogical : public IServerLogical
{
public:
    CServerLogical();
    ~CServerLogical();
public:
    bool InitCfg(const string& strFileName);

public:
    virtual bool Start();
    virtual void Stop();
    virtual void Update();
    virtual void DoIdleTask();
   
    virtual int32 OnHttpClientRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote,SHttpRequestPara_ptr pReq,SHttpResponsePara_ptr pRes);
    virtual int32 OnUDPMessage(CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds, IN int thread_index,uint8 algo);
    virtual int32 OnTCPMessage(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_id, CDataStream& recvds,CDataStream& sendds);
    virtual int32 OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& hiRemote,CDataStream& sendds);
    virtual int32 OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& hiRemote);
protected:
    int32 HandleReqMsg(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds,CDataStream& sendds);
    int32 HandleRespMsg(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds,CDataStream& sendds);
    int32 HandleNotifyMsg(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds,CDataStream& sendds);
private:
    bool OnConnect(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
    bool OnDisconnect(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
    bool OnStatusReport(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
    bool OnPlayReq(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
    bool OnPauseReq(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
    bool OnCmdReq(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
private:
    bool OnPlayResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds);
    bool OnPauseResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds);
    bool OnMediaCmdResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds);
    bool OnCloseResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds);
    bool OnFrameNotify(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
    bool OnEosNotify(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds);
private:
    int OnHttpRootRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pRes);
    int OnHttpFaviconRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pRes);
    int OnHttpCloudStorage(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pRes);
    int OnHttpLiveRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pRes);
    int OnHttpDumpInfoRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pRes);
private:
	void InitLoginStatus();
	void UpdateLoadReport();
public:
    static CServerLogical* GetLogical(){return &logic_;}
public:
    CStreamCfg_ptr GetServCfg(){return pSevrCfg_;}
    CTokenMgr_ptr GetTokenMgr(){return pTokenMgr_;}
    CMediaSessionMgr_ptr GetMediaSessionMgr(){return pMediaSessionMgr_;}
    CStreamStatusReport_ptr GetStreamStatusReport(){return pStreamStatusReport_;}
private:
    static CServerLogical logic_;
private:
	tick_t tick_start_;
	protocol::StsLoginReq		login_req_;
	protocol::StsLoadReportReq	load_report_;
private:
    CStreamCfg_ptr			pSevrCfg_;
    CTokenMgr_ptr			pTokenMgr_;
	CMediaSessionMgr_ptr	pMediaSessionMgr_;
	CSysMonitorThread_ptr	pSysMonitorThread_;
	CStreamStatusReport_ptr pStreamStatusReport_;
};

#endif //__SERVER_LOGICAL_H__

