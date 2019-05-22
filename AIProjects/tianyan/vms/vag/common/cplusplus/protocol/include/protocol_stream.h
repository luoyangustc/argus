
#ifndef __PROTOCOL_STREAM_H__
#define __PROTOCOL_STREAM_H__

#include "protocol_header.h"

namespace protocol{
    
    // 媒体方向
    enum MediaDirectType
    {
        MEDIA_DIR_SEND_ONLY  = 0x01,    // 仅作为发送端
        MEDIA_DIR_RECV_ONLY  = 0x02,    // 仅作为接受端
        MEDIA_DIR_SEND_RECV  = 0x03,    // 既作为发送端，也接受
    };

    // 帧类型
    #define  MediaFrameMask 0x7F    //帧掩码
    #define  IsExpireFrame(x)   ( ((x)&~MediaFrameMask)!=0 )                // 是否是过期帧
    #define  IsAudioFrame(x)    ( ((x)&MediaFrameMask)==FRAME_TYPE_AU )     // 是否是音频帧
    #define  IsKeyFrame(x)      ( ((x)&MediaFrameMask)==FRAME_TYPE_I )      // 是否是关键帧

    // 媒体命令
    enum MediaCMDType
    {
        MEDIA_CMD_VIDEO_OPEN    = 0x0001,    //视频打开
        MEDIA_CMD_VIDEO_CLOSE   = 0x0002,    //视频关闭
        MEDIA_CMD_AUDIO_OPEN    = 0x0003,    //音频打开
        MEDIA_CMD_AUDIO_CLOSE   = 0x0004,    //音频关闭
    };
    
    /*************************************
    **类型：媒体连接接口(MSG_ID_MEDIA_CONNECT)
    **定义: StreamMediaConnectReq
    **方向：Device->STREAM or CU->STREAM
    **************************************/
    struct StreamMediaConnectReq
	{
		uint32 mask;
        
        //0x01 基本会话信息
        uint16 session_type;    // refer to 'MediaSessionType'
        string session_id;      // 会话ID
		uint8 session_media;	// 媒体类型：0x01:Video 0x02:Audio, 0x03:all
        string endpoint_name;   // Device为设备ID, CU为登陆账号, StreamServ固定为"UlucuStream"
        uint16 endpoint_type;   // 1:Device 2:CU 3:StreamServ(Relay机制中使用) refer to 'EndPointType'
		string device_id;
        uint16 channel_id;
        uint8  stream_id;       // refer to 'EnSteamType'
        token_t token;          // 校验token

        //0x02 视频信息
        uint8 video_direct;     // 0x01:sendonly, 0x02:recvonly, 0x03:sendrecv        
        VideoCodecInfo video_codec;
        
        
        //0x04 音频信息
        uint8 audio_direct;     // 0x01:sendonly, 0x02:recvonly, 0x03:sendrecv
        AudioCodecInfo audio_codec;
        
        //0x08 媒体时间
        uint32 begin_time;      // 开始时间(实时浏览时为0), UTC时间
        uint32 end_time;        // 结束时间(实时浏览时为0), UTC时间
        
        //0x10 媒体路由信息(Relay机制中使用)
		uint8 route_table_size;
        vector<HostAddr> route_tables;

        //0x20 用户自定义信息
        string user_info;
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaConnectReq& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaConnectReq& req);
    
    struct StreamMediaConnectResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaConnectResp& resp);
    CDataStream& operator>>(CDataStream& ds, StreamMediaConnectResp& resp);
    
    /*************************************
    **类型：媒体断开接口(MSG_ID_MEDIA_DISCONNECT)
    **定义: StreamMediaDisconnectReq
    **方向：Device->STREAM or CU->STREAM
    **************************************/
    struct StreamMediaDisconnectReq
	{
		uint32 mask;
        
        //0x01 基本会话信息
        string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaDisconnectReq& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaDisconnectReq& req);
    
    struct StreamMediaDisconnectResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaDisconnectResp& resp);
    CDataStream& operator>>(CDataStream& ds, StreamMediaDisconnectResp& resp);
    
    /*************************************
    **类型：媒体播放接口(MSG_ID_MEDIA_PLAY)
    **定义: StreamMediaPlayReq
    **方向：CU->STREAM or STREAM->Device
    **************************************/
    struct StreamMediaPlayReq
	{
        uint32 mask;
        
        //0x01 基本会话信息
        string session_id;      // 会话ID        
        
        //0x02 时间戳(回放拖拽使用, 没有该字段表示从文件当前位置播放)
        uint32 begin_time;      // 开始时间, UTC时间
        uint32 end_time;        // 结束时间(0表示一直播放到结尾)
        
        //0x04 倍速信息(回放/下载控制速率)
        uint8 speed;            // 倍速(1~16), 正常播放速度为1, 
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaPlayReq& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaPlayReq& req);
    
    struct StreamMediaPlayResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaPlayResp& resp);
    CDataStream& operator>>(CDataStream& ds, StreamMediaPlayResp& resp);
    
    /*************************************
    **类型：媒体暂停接口(EN_CMD_MEDIA_PAUSE)
    **定义: StreamMediaPause
    **方向：CU->STREAM or STREAM->Device
    **************************************/
    struct StreamMediaPauseReq
	{
        uint32 mask;
        
        //0x01 基本会话信息
        string session_id;      // 会话ID
        
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaPauseReq& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaPauseReq& req);
    
    struct StreamMediaPauseResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaPauseResp& resp);
    CDataStream& operator>>(CDataStream& ds, StreamMediaPauseResp& resp);
    
    /*************************************
    **类型：媒体控制接口(MSG_ID_MEDIA_CMD)
    **定义: StreamMediaCmdReq
    **方向：CU->STREAM or STREAM->Device
    **************************************/
    struct StreamMediaCmdReq
	{
		uint32 mask;
        
        //0x01 基本会话信息
        string session_id;  // 会话ID
        uint8 cmd_type;     // refer to 'MediaCMDType'
        
        //0x02 命令参数
        uint16 param_data_size;
        vector<uint8> param_datas;
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaCmdReq& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaCmdReq& req);
    
    struct StreamMediaCmdResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaCmdResp& resp);
    CDataStream& operator>>(CDataStream& ds, StreamMediaCmdResp& resp);

    /*************************************
    **类型：媒体状态上报接口(MSG_ID_MEDIA_STATUS)
    **定义: StreamMediaStatus
    **方向：CU->STREAM or Device->STREAM
    **************************************/
    struct StreamMediaStatusReq
	{
		uint32 mask;
        
        //0x01 基本会话信息
        string session_id;      // 会话ID
        
        //0x02 视频状态
        uint8 video_status;     // 1:on 2:off
        
        //0x04 音频状态
        uint8 audio_status;     // 1:on 2:off
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaStatusReq& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaStatusReq& req);
    
    struct StreamMediaStatusResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaStatusResp& resp);
    CDataStream& operator>>(CDataStream& ds, StreamMediaStatusResp& resp);
    
    /******************************************
    **类型：媒体帧数据通知(MSG_ID_MEDIA_FRAME)
    **定义: StreamMediaFrameNotify
    **方向：Device->STREAM or STREAM->CU
    ******************************************/
    struct StreamMediaFrameNotify
	{
		uint32 mask;
        
        //0x01 基本会话信息
        string session_id;      // 会话ID
        
        //0x02
		uint8 frame_type;	    // 0x01:I帧 0x02:P帧 0x03:音频帧, 最高位(第8位)置1表示过期帧
        uint32 frame_av_seq;    // 音视频帧序号(音频+视频总和)
		uint32 frame_seq;       // 帧序号(音频或视频帧序号)
        uint32 frame_base_time; // 帧基准时间戳(单位s)
		uint32 frame_ts;        // 帧时间戳(单位ms)
		uint32 frame_size;      // 帧大小

		//0X04
		uint32 crc32_hash;

		//0x08
		uint32 offset;
		uint32 data_size;
		vector<uint8> datas;      // 可变长, 等于data_size
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaFrameNotify& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaFrameNotify& req);
    
    /*************************************
    **类型：媒体EOS通知(MSG_ID_MEDIA_EOS)
    **定义: StreamMediaEosNotify
    **方向：Device->STREAM or STREAM->CU
    **************************************/
    struct StreamMediaEosNotify
	{
		uint32 mask;
        
        //0x01 基本会话信息
        string session_id;  // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaEosNotify& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaEosNotify& req);
    
    /*************************************
    **类型：媒体关闭请求(EN_CMD_MEDIA_CLOSE)
    **定义: StreamMediaClose
    **方向：STREAM->CU or STREAM->Device
    **************************************/
    struct StreamMediaCloseReq
	{
		uint32 mask;
        
        //0x01 基本会话信息
        string session_id;  // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaCloseReq& req);
    CDataStream& operator>>(CDataStream& ds, StreamMediaCloseReq& req);
    
    struct StreamMediaCloseResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, StreamMediaCloseResp& resp);
    CDataStream& operator>>(CDataStream& ds, StreamMediaCloseResp& resp);
};

#endif

