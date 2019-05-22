#ifndef __CU_STREAM_H__
#define __CU_STREAM_H__

#include "CommonInc.h"
#include "media_info.h"

enum EnCuStreamStatus
{
    en_cu_stream_init = 0,
    en_cu_stream_connected,
    en_cu_stream_play,
    en_cu_stream_pause,
    en_cu_stream_close,
    en_cu_stream_disconnect,
    en_cu_stream_error
};

class CCuStream
{
public:
    CCuStream();
    ~CCuStream();

    CHostInfo GetRemote(){return hi_remote_;}
    string GetUserName(){return user_name_;}
    EnMediaSessionType GetSessionType(){return session_type_;}
    EnCuStreamStatus GetStatus(){return status_;}
    EndPointType GetEndpointType(){return endpoint_type_;}

    bool IsAudioOpen(){return audio_open_;}
    bool IsVideoOpen(){return video_open_;}
    bool IsAlive();
    bool IsConnected();

    void Update();
    bool Close();
    bool OnTcpClose();

    bool OnStream(MI_FrameData_ptr frame_data);
public:
    bool OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp);
    bool OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp);
    bool OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp);
    bool OnPlayReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayReq& req, StreamMediaPlayResp& resp);
    bool OnPauseReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseReq& req, StreamMediaPauseResp& resp);
    bool OnCmdReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdReq& req, StreamMediaCmdResp& resp);

private:
    bool SendMsg(const char* msg, uint32 length);

private:
    ITCPSessionSendSink* send_sink_;
    CHostInfo hi_remote_;
    string user_name_;
    EndPointType endpoint_type_;

    EnCuStreamStatus status_;
    uint32 send_seq_;

    bool running_;
    EnMediaSessionType session_type_;
    string session_id_;
    SDeviceChannel dc_;
    bool video_open_;
    bool audio_open_;

    token_t token_;
    uint32 token_flag_;

    protocol::VideoCodecInfo video_info_;
    uint8 video_direct_;

    protocol::AudioCodecInfo audio_info_;
    uint8 audio_direct_;

    uint32 begin_time_;
    uint32 end_time_;

    tick_t last_active_tick_;
};

typedef boost::shared_ptr<CCuStream> CCuStream_ptr;

#endif //__CU_STREAM_H__
