#ifndef __DEVICE_MGR_H__
#define __DEVICE_MGR_H__

#include <map>
#include <vector>
#include <sstream>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/shared_ptr.hpp>
#include "base/include/DeviceID.h"
#include "base/include/tick.h"
#include "protocol/include/protocol_device.h"
#include "protocol/include/protocol_status.h"
#include "DeviceContext.h"

using namespace std;
using namespace protocol;

class CDeviceMgr
{
public:
	CDeviceMgr();
	~CDeviceMgr(void);
    void Update();
	void DoIdleTask();
    bool OnTCPClosed( const CHostInfo& hiRemote );
    ostringstream& DumpInfo(ostringstream& oss, string& verbose);
    ostringstream& DumpDeviceInfo(const string& did,ostringstream& oss);
public:
    uint32 GetDeviceNum();
    uint32 GetConnectNum();
    CDeviceContext_ptr GetDeviceContext(const string& did);
    CDeviceContext_ptr GetDeviceContext(const CHostInfo& hiRemote);
    void GenDeviceStatus(vector<SDeviceSessionStatus>& deviceStatus);
public:
    //bool MediaOpen(const SDeviceChannel& dc, const string& session_id, uint16 session_type, const SMediaDesc& desc, const vector<HostAddr>& addrs);
    //bool MediaClose(const SDeviceChannel& dc, const string& session_id);
public:
    bool OnDeviceOffline(const CHostInfo& hiRemote, const string& did);
    bool OnDeviceOnline(const CHostInfo& hiRemote, const string& did, CDeviceContext_ptr pDeviceCtx);
public:
    bool ON_DeviceLoginRequest(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceLoginReq& req, DeviceLoginResp& resp);
	bool ON_DeviceAbilityReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceAbilityReportReq& req );
    bool ON_StatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceStatusReportReq& report, DeviceStatusReportResp& resp);
    bool ON_DeviceAlarmReport(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,const DeviceAlarmReportReq& report, DeviceAlarmReportResp& resp);
    bool ON_DeviceMediaOpenResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceMediaOpenResp& resp);
    bool ON_DeviceSnapResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceSnapResp& resp);
private:
	boost::recursive_mutex lock_;
	map<CHostInfo,CDeviceContext_ptr > hi_devices_;
	map<string, CDeviceContext_ptr > did_devices_;
};

typedef boost::shared_ptr<CDeviceMgr> CDeviceMgr_ptr;

#endif //__DEVICE_MGR_H__
