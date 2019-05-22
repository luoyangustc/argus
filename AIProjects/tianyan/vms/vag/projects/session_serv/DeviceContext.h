#ifndef __DEVICE_CONTEXT_H__
#define __DEVICE_CONTEXT_H__

#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include "base/include/HostInfo.h"
#include "base/include/DeviceChannel.h"
#include "base/include/GetTickCount.h"
#include "netlib_framework/include/ITCPSessionSendSink.h"
#include "protocol/include/protocol_device.h"
#include "protocol/include/protocol_client.h"
#include "protocol/include/protocol_stream.h"
#include "MediaSession.h"
#include "Snaper.h"

using namespace std;
using namespace protocol;

class IDeviceSnapSink
{
public:
    virtual void OnDeviceSnapAck(int err_code, const string& err_msg, const string& pic_url="") = 0;
};

typedef boost::shared_ptr<IDeviceSnapSink> IDeviceSnapSink_ptr;

class CDeviceContext
{
public:
    CDeviceContext();
    ~CDeviceContext();
    void Update();
    bool IsAlive();
    void OnTcpClose(const CHostInfo& hiRemote);
    ostringstream& DumpInfo(ostringstream& oss, string verbose = "true");
public:
    CHostInfo GetRemote(){ return hi_remote_; }
    string GetDeviceId(){ return device_id_; }
    protocol::STimeVal GetLoginTimestamp(){ return login_timestamp_; }
    void GetVersion( string& ver );
    uint8 GetDeviceType();
    uint8 GetChannelNum();
    bool GetChannel( uint16 channel_id,  protocol::DevChannelInfo& channel_info );
    void GetChannels( vector<protocol::DevChannelInfo>& channels );
    void GetAddrHostInfo(CHostInfo& stream_addr, CHostInfo& remote_public_addr);
public:
    bool MediaOpen(SDeviceChannel dc, string session_id, uint16 session_type, SMediaDesc desc, vector<HostAddr> addrs);
    bool MediaClose(SDeviceChannel dc, string session_id);
    bool Screenshot( const string& device_id, uint16 channel_id, IDeviceSnapSink_ptr user_sink );
    bool PtzCtrl(const string& device_id, uint16 channel_id, uint16 cmd_type);
    bool MgrUpdate(const string& device_id, uint16 channel_id, uint16 mgr_type);
public:
    bool ON_DeviceLoginRequest(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const DeviceLoginReq& req, DeviceLoginResp& resp);
	bool ON_DeviceAbilityReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceAbilityReportReq& req);
    bool ON_StatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceStatusReportReq& report, DeviceStatusReportResp& resp);
    bool ON_DeviceMediaOpenResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceMediaOpenResp& resp);
    bool ON_DeviceSnapResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceSnapResp& resp);
private:
    bool SendMessage(uint8* data_buff, uint32 data_size);
    bool OnChanenlStatusUpate();
private:
    boost::recursive_mutex lock_;
    boost::atomic_uint cmd_seq_;
    ITCPSessionSendSink* send_sink_;

    CHostInfo hi_remote_;
    CHostInfo hi_private_addr_;
    
    string device_id_;
    string version_;
    uint8 dev_type_;
    uint16 channel_num_;
    vector<protocol::DevChannelInfo> channels_;
    token_t token_;

    uint32 disc_size_;
    uint32 disc_free_size_;

    uint8 sdcard_status_;

    uint8 media_trans_type_;
    uint8 max_live_streams_per_ch_;
    uint8 max_playback_streams_per_ch_;
    uint8 max_playback_streams_;

    protocol::STimeVal login_timestamp_;
    tick_t last_active_tick_;

    map< uint16, CSnaper_ptr > snap_tasks_;
    map< uint16, vector<IDeviceSnapSink_ptr> > snap_sinks_;
};

typedef boost::shared_ptr<CDeviceContext> CDeviceContext_ptr;

#endif //__SESSION_CONTEXT_H__
