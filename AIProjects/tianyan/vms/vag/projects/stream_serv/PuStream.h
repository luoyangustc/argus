#ifndef __PU_STREAM_H__
#define __PU_STREAM_H__

#include "CommonInc.h"
#include "FrameMgr.h"
#include "variant.h"

enum EnPuStreamStatus
{
    en_pu_stream_init = 0,
    en_pu_stream_connected,
    en_pu_stream_play,
    en_pu_stream_pause,
    en_pu_stream_closing,
    en_pu_stream_disconnect,
    en_pu_stream_error
};

class CPuStream
{
public:
    CPuStream();
    ~CPuStream();

    bool IsAudioOpen(){return audio_open_;}
    bool IsVideoOpen(){return video_open_;}
    bool IsAlive();
    bool IsConnected();

    void Update();

    bool Play();
    bool Pause();
    bool AudioCtrl(bool onoff);
    bool VideoCtrl(bool onoff);
    bool Close();
    bool OnTcpClose();

    void DumpInfo(Variant& info);

    bool GetFrameData(uint32 frm_seq, uint8 frm_type, MI_FrameData_ptr& frmData);
    bool GetRecentData(stack<MI_FrameData_ptr>& frm_datas);

public:
    //device request
    bool OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp);
    bool OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp);
    bool OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp);
    
    //device response
    bool OnPlayResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayResp& resp);
    bool OnPauseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseResp& resp);
    bool OnCmdResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdResp& resp);

    //device notify
    bool OnFrameNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaFrameNotify& notify);
public:
    CHostInfo GetRemote(){return hi_remote_;}
    string GetEndpointName(){return endpoint_name_;}
    EnMediaSessionType GetSessionType(){return session_type_;}
    uint8 GetSessionMedia(){return session_media_;}
    SDeviceChannel GetDeviceChannel(){ return dc_;}
    int GetAudioInfo(protocol::AudioCodecInfo& audio_info);
    int GetVideoInfo(protocol::VideoCodecInfo& video_info);
    EnPuStreamStatus GetStatus();
private:
	bool SendMsg(const char* msg, uint32 length);
private:
    boost::recursive_mutex lock_;
    ITCPSessionSendSink* send_sink_;
    CHostInfo hi_remote_;
    string endpoint_name_;

    int status_; //EnPuStreamStatus
    boost::atomic_uint32_t send_seq_;

    bool running_;
    bool video_open_;
    bool audio_open_;

    EnMediaSessionType session_type_;
    string session_id_;
    uint8 session_media_;
    SDeviceChannel dc_;

    token_t token_;

    CFrameMgr_ptr frame_mgr_ptr_;

    uint8 video_type_;
    uint8 video_direct_;
    protocol::VideoCodecInfo video_info_;

    uint8 audio_direct_;
    bool has_audio_info_;
    protocol::AudioCodecInfo audio_info_;

    uint32 begin_time_;
    uint32 end_time_;

    tick_t last_active_tick_;
    tick_t last_recv_tick_;
    tick_t last_audio_tick_;
    tick_t last_video_tick_;
};

typedef boost::shared_ptr<CPuStream> CPuStream_ptr;

#endif