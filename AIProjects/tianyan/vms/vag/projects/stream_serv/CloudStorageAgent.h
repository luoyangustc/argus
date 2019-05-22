#ifndef _CLOUD_STORAGE_AGENT_H
#define _CLOUD_STORAGE_AGENT_H
#include <queue>
#include "CommonInc.h"
#include "BufferInfo.h"
#include "media_info.h"
#include "protocol_stream.h"

class CCloudStorageAgent : public ITCPClientSink
{
    enum en_serv_agent_status
    {
        en_serv_agent_init = 0,
        en_serv_agent_connecting,
        en_serv_agent_connected,
        en_serv_agent_login,
        en_serv_agent_runloop,
        en_serv_agent_error
    };
public:
    virtual int OnTCPConnected(uint32 ip, uint16 port);
    virtual int OnTCPConnectFailed(uint32 ip, uint16 port);
    virtual int OnTCPClose(uint32 ip, uint16 port);
    virtual int OnTCPMessage(uint32 ip, uint16 port, uint8* data, uint32 data_len);
public:
	CCloudStorageAgent(const CHostInfo& hiServer);
	~CCloudStorageAgent(void);
    void Update();
    void SetDC(const SDeviceChannel& dc);
    void SetVideoInfo(protocol::VideoCodecInfo video_info);
    void SetAudioInfo(protocol::AudioCodecInfo audio_info);
    int OnStream(MI_FrameData_ptr frame_data);
private:
    int StartConnect();
    int SendMediaConnectReq();
    int SendMediaFrameNotify();
    int OnMediaConnectResp(const StreamMediaConnectResp& resp);
private:
    boost::recursive_mutex lock_;
    ITCPClient* agent_;
	CHostInfo hi_server_;
    int agent_status_;
    uint32 agent_status_tick_;
    uint32 send_seq_;
    std::queue<MI_FrameData_ptr> send_msg_que_;
private:
    SDeviceChannel dc_;
    int storage_type_;
    protocol::VideoCodecInfo video_info_;
    protocol::AudioCodecInfo audio_info_;
};

typedef boost::shared_ptr<CCloudStorageAgent> CCloudStorageAgent_ptr;

#endif  //_CLOUD_STORAGE_AGENT_Hrecv