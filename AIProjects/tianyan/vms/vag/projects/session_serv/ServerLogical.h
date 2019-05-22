#ifndef __SERVER_LOGICAL_H__
#define __SERVER_LOGICAL_H__

#include <list>
#include <string>
#include <vector>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/shared_array.hpp>
#include "IServerLogical.h"
#include "protocol_status.h"
#include "common_thread_group.h"
#include "MediaSessionMgr.h"
#include "DeviceMgr.h"
#include "UserMgr.h"
#include "TokenMgr.h"
#include "StatusReportClient.h"
#include "global_configitem.h"
#include "MonitorThread.h"
#include "ServerConfig.h"
//#include "GB28181DeviceMgr.h"

using namespace std;
using namespace protocol;

class CServerLogical;
CServerLogical* GetService();

class CServerLogical : public IServerLogical
{
public:
    CServerLogical();
    ~CServerLogical();
	bool InitCfg(const string& strFileName);
public:
    virtual bool Start();
    virtual void Stop();
    virtual int32 OnHttpClientRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote,SHttpRequestPara_ptr pReq,SHttpResponsePara_ptr pRes);
    virtual int32 OnUDPMessage(CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds, IN int thread_index){return -1;}
    virtual int32 OnTCPMessage(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_id, CDataStream& recvds,CDataStream& sendds);
    virtual int32 OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& hiRemote,CDataStream& sendds);
    virtual int32 OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& hiRemote);

    virtual void Update();
    virtual void DoIdleTask();
public:
	void GetServerHostAddr(string& ip, uint16& port);
    const char* GetStreamScheduler() {return pServCfg_->GetGlsb().c_str();}
    bool IsEnableTokenAuth() {return pServCfg_->IsTokenCheck();}
    const string& GetAccessKey() {return access_key_;}
    void UpdateLoadReport();
private:
    //device message handle
    bool ON_DeviceLoginRequest(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds);
	bool ON_DeviceAbilityReport(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds);
    bool ON_StatusReport(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds);
    bool ON_DeviceAlarmReport(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds);
    bool ON_DeviceMediaOpenResp(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds);
    bool ON_DeviceSnapResp(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds);
private:
	//client message handle
	bool ON_CuLoginRequest(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds);
	bool ON_CuStatusReport(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds);
    bool ON_CuMediaOpen(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds);
    bool ON_CuMediaClose(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds);
private:
    int OnHttpRootRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp);
    int OnHttpFaviconRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp);
    int OnHttpSnapRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp);
    int OnHttpPtzctrlRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp);
    int OnHttpCloudStorage(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp);
    int OnHttpLiveRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp);
    int OnHttpDeviceMgrUpdate(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp);
private:
    char self_ip_[256];
    uint16 self_port_;
	bool has_private_ip_;

	CServerCfg_ptr pServCfg_;
    CStatusReportClient_ptr pStatusReportClient_;
	CDeviceMgr_ptr pDeviceMgr_;
	CUserMgr_ptr pUserMgr_;	
	CTokenMgr_ptr pTokenMgr_;
    CMediaSessionMgr_ptr pMediaSessionMgr_;
    //CGB28181DeviceMgr_ptr pGB28181DeviceMgr_;
	
    protocol::StsLoginReq login_req_;
    protocol::StsLoadReportReq load_report_;

	tick_t start_tick_;
    tick_t last_update_load_tick_;
	
	list<uint32> local_ips_;
	uint32 local_first_ip_;
    string stream_scheduler_;

    bool enable_token_auth_;
    string access_key_;
    string secret_key_;

	boost::recursive_mutex common_lock_;
	CCommon_Thread_Group_ptr pCommonThreadGroup_;
    CSysMonitorThread_ptr pSysMonitorThread_;
    GlobalConfigItem global_configitem_;
public:
	CServerCfg_ptr GetServCfg(){return pServCfg_;}
    CUserMgr_ptr GetUserMgr(){return pUserMgr_;}
    CDeviceMgr_ptr GetDeviceMgr(){return pDeviceMgr_;}
    //CGB28181DeviceMgr_ptr GetGB28181DeviceMgr(){return pGB28181DeviceMgr_;}
    CTokenMgr_ptr GetTokenMgr(){return pTokenMgr_;}
    CMediaSessionMgr_ptr GetMediaSessionMgr(){return pMediaSessionMgr_;}
    CStatusReportClient_ptr GetStatusReportClient() {return pStatusReportClient_;}
    const GlobalConfigItem& GetGlobalConfigItem() { return global_configitem_; }

    CDeviceContext_ptr  GetDeviceContext(const CHostInfo& hi );
    CDeviceContext_ptr  GetDeviceContext(const string& device_id );
    CUserContext_ptr    GetUserContext(const CHostInfo& hi );
    void                GetUserContext(const string& user_name, OUT vector<CUserContext_ptr>& user_contexts );

    MediaSession_ptr    GetMediaSession(const string& session_id);
    MediaSession_ptr    GetMediaSession(const SDeviceChannel& dc, SessionType session_type);
    
private:
    static CServerLogical logic_;
public:
    static CServerLogical* GetLogical(){return &logic_;}
};

#endif //__SERVER_LOGICAL_H__
