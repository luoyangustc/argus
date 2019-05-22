#ifndef __PROTOCOL_HEADER_H__
#define __PROTOCOL_HEADER_H__

#include "vos_types.h"

#undef  EXT
#ifndef __PROTOCOL_HEADER_C__
#define EXT extern
#else
#define EXT
#endif

#define MAX_MSG_BUFF_SIZE	        (16*1024)
#define MAX_TCP_PACKET_SIZE		    (64*1024)
#define MAX_IP_LEN                  (128)

enum EnMsgType
{
    MSG_TYPE_REQ    = 0x00000001,   //request msg
    MSG_TYPE_RESP   = 0x00000002,   //response msg
    MSG_TYPE_NOTIFY = 0x00000003    //notify msg
};

enum EnMsgID
{
    //Exchange key
    MSG_ID_EXCHANGE_KEY             = 0x00000001,

    //Device->SMS
    MSG_ID_DEV_LOGIN                = 0x01000001,
    MSG_ID_DEV_ABILITY_REPORT       = 0x01000002,
    MSG_ID_DEV_STATUS_REPORT        = 0x01000003,
    MSG_ID_DEV_ALARM_REPORT         = 0x01000004,
    MSG_ID_DEV_PIC_UPLOAD_REPORT    = 0x01000005,

    //SMS->Device
    MSG_ID_DEV_MEDIA_OPEN           = 0x02000001,
    MSG_ID_DEV_MEDIA_CLOSE          = 0x02000002,

    //CU->SMS
    MSG_ID_CU_LOGIN                 = 0x03000001,
    MSG_ID_CU_STATUS_REPORT         = 0x03000002,
    MSG_ID_CU_MEDIA_OPEN            = 0x03000003,
    MSG_ID_CU_MEDIA_CLOSE           = 0x03000004,

    //CU->SMS->Device
    MSG_ID_DEV_SNAP                 = 0x04000001,
    MSG_ID_DEV_CTRL                 = 0x04000002,
    MSG_ID_DEV_PARAM_SET            = 0x04000003,
    MSG_ID_DEV_PARAM_GET            = 0x04000004,
    MSG_ID_DEV_RECORD_LIST_QUERY    = 0x04000005,

    //Stream
    MSG_ID_MEDIA_CONNECT            = 0x05000001,   // CU->Stream, Device->Stream
    MSG_ID_MEDIA_DISCONNECT         = 0x05000002,   // CU->Stream, Device->Stream
    MSG_ID_MEDIA_STATUS             = 0x05000003,   // CU->Stream, Device->Stream
    MSG_ID_MEDIA_PLAY               = 0x05010001,   // CU->Stream, Stream->Device
    MSG_ID_MEDIA_PAUSE              = 0x05010002,   // CU->Stream, Stream->Device
    MSG_ID_MEDIA_CMD                = 0x05010003,   // CU->Stream, Stream->Device
    MSG_ID_MEDIA_FRAME              = 0x05020001,   // Device->Stream, Stream->CU
    MSG_ID_MEDIA_EOS                = 0x05020002,   // Device->Stream, Stream->CU
    MSG_ID_MEDIA_CLOSE              = 0x05030001,   // Stream->Device, Stream->CU


    //SMS->STATUS, STREAM->STATUS
    MSG_ID_STS_LOGIN                    = 0x06000001,
    MSG_ID_STS_LOAD_REPORT              = 0x06000002,
    MSG_ID_STS_SESSION_STATUS_REPORT    = 0x06000003,
    MSG_ID_STS_STREAM_STATUS_REPORT     = 0x06000004
};

enum EnRespCode
{
    //for all
    EN_SUCCESS                      =    0,

    //for all
    EN_ERR_START    = -1000,
    EN_ERR_SERVICE_UNAVAILABLE,     // 服务不可用
    EN_ERR_MALLOC_FAIL,             // 服务不可用
    EN_ERR_MSG_PARSER_FAIL,         // 消息解析错误      
    EN_ERR_TOKEN_CHECK_FAIL,        // token校验错误
    EN_ERR_DEVICE_OFFLINE,          // 设备不在线
    EN_ERR_CHANNEL_OFFLINE,         // 设备不在线
    EN_ERR_ENDPOINT_UNKWON,         // 节点类型未知
    EN_ERR_NOT_SUPPORT,             // 不支持的业务等

    //for device
    EN_DEV_ERR_GET_STREAM_SERV_FAIL     = -600,
    EN_DEV_ERR_CONNECT_STREAM_REPEAT,
    EN_DEV_ERR_CREATE_STREAM_FAIL,
    EN_DEV_ERR_STREAM_DISCONNECT,
    EN_DEV_ERR_STREAM_DIFFERENT,
    EN_DEV_ERR_CONNECT_STREAM_FAIL,
    EN_DEV_ERR_APP_CB_FAIL,
    EN_DEV_ERR_IN_BUSY,
    EN_DEV_ERR_TIMEOUT,

    //for client
    EN_CU_ERR_GET_STREAM_SERV_FAIL     = -300,
    EN_CU_ERR_CONNECT_STREAM_REPEAT,
    EN_CU_ERR_CREATE_STREAM_FAIL,
    EN_CU_ERR_STREAM_DISCONNECT
};

enum EndPointType
{
    EP_UNKNOWN      = 0x00,
    EP_DEV   	    = 0x01,
    EP_CU           = 0x02,
    EP_STREAM       = 0x03,
    EP_SMS          = 0x04
};

// 设备类型定义
enum EnDeviceType
{
    DEV_TYPE_IPC     = 0,
    DEV_TYPE_NVR     = 1,
    DEV_TYPE_DVR     = 2,
    DEV_TYPE_QNSMB   = 3     /* 七牛智能盒子（QiNiu SmartBox）*/
};

/* 帧类型定义 */
enum EnFrameType
{
    FRAME_TYPE_I    = 0,    // I帧
    FRAME_TYPE_P    = 1,    // P帧
    FRAME_TYPE_AU   = 2     // 音频帧
};

// 视频编码类型定义
enum EnVideoEncodeType
{
    VIDEO_H264  = 0,
    VIDEO_H265  = 1,
};

// 音频编码类型定义
enum EnAudioEncodeType
{
    AUDIO_AAC               = 0,
    AUDIO_G711_A            = 1,
    AUDIO_G711_U            = 2,
    AUDIO_MP3               = 3,
};

// 音频通道类型定义
enum EnAudioChannelType
{
    AUDIO_CH_MONO           = 0,    // 单声道
    AUDIO_CH_STEREO         = 1,    // 立体声道
};
#define  IsAudioCH_Mono(x)      ( (x)==AUDIO_CH_MONO )      // 是否是单声道
#define  IsAudioCH_Stereo(x)    ( (x)==AUDIO_CH_STEREO )    // 是否是立体声道

// 音频位宽类型定义
enum EnAudioBitwidthType
{
    AUDIO_BW_8BIT           = 0,    //位宽：8bit
    AUDIO_BW_16BIT          = 1,    //位宽：16bit
};
#define  IsAudioBW_8Bit(x)  ( (x)==AUDIO_BW_8BIT )      //是否是8bit位宽
#define  IsAudioBW_16Bit(x) ( (x)==AUDIO_CH_STEREO )    //是否是16bit位宽

// 音频采样率类型定义
enum EnAudioSampleRateType
{
    AUDIO_SR_8_KHZ       = 0,    // 采样率：8khz
    AUDIO_SR_11_025_KHZ  = 1,    // 采样率：11.025khz
    AUDIO_SR_12_KHZ      = 2,    // 采样率：12khz
    AUDIO_SR_16_KHZ      = 3,    // 采样率：16khz
    AUDIO_SR_22_05_KHZ   = 4,    // 采样率：22.05khz
    AUDIO_SR_24_KHZ      = 5,    // 采样率：24khz
    AUDIO_SR_32_KHZ      = 6,    // 采样率：32khz
    AUDIO_SR_44_1_KHZ    = 7,    // 采样率：44.1khz
    AUDIO_SR_48_KHZ      = 8,    // 采样率：48khz
    AUDIO_SR_64_KHZ      = 9,    // 采样率：64khz
    AUDIO_SR_88_2_KHZ    = 10,   // 采样率：88.2khz
    AUDIO_SR_96_KHZ      = 11    // 采样率：96khz
};

// 码流类型定义
enum EnStreamType
{
    STREAM_TYPE_MAIN        = 0,    // 主码流
    STREAM_TYPE_SUB01       = 1,    // 子码流1
    STREAM_TYPE_SUB02       = 2,    // 子码流2
};

/* 通道状态定义 */
enum EnChannelStatus
{
    CHANNEL_STS_OFFLINE     = 0,    // 通道状态下线
    CHANNEL_STS_ONLINE      = 1,    // 通道状态上线
};

// 告警类型定义
enum EnAlarmType
{
    //平台告警类型定义
    ALARM_TYPE_DEVICE_ONLINE    = 0x00000001,   // 设备上线
    ALARM_TYPE_DEVICE_OFFLINE   = 0x00000002,   // 设备下线

    //设备端告警类型定义
    ALARM_TYPE_MOTION_DETECT    = 0x00010001,   // 移动侦测
    ALARM_TYPE_VIDEO_LOST       = 0x00010002,   // 视频丢失
    ALARM_TYPE_VIDEO_SHELTER    = 0x00010003,   // 视频遮挡	
    ALARM_TYPE_DISC             = 0x00010004,   // 磁盘告警
    ALARM_TYPE_RECORD           = 0x00010005,   // 录像告警
};

// 告警状态定义
enum EnAlarmStatus
{
    ALARM_STS_CLEAN      = 0,   // 告警消去
    ALARM_STS_ACTIVE     = 1,   // 告警开启
    ALARM_STS_KEEP       = 2,   // 告警保持
};
enum MediaSessionType
{
    MEDIA_SESSION_TYPE_LIVE                 = 0x0001,
    MEDIA_SESSION_TYPE_PU_PLAYBACK          = 0x0002,
    MEDIA_SESSION_TYPE_PU_DOWNLOAD          = 0x0003,
    MEDIA_SESSION_TYPE_DIRECT_LIVE          = 0x8001,
    MEDIA_SESSION_TYPE_DIRECT_PU_PLAYBACK   = 0x8002,
    MEDIA_SESSION_TYPE_DIRECT_PU_DOWNLOAD   = 0x8003
};


#pragma pack(1)

typedef struct MsgHeader
{
    vos_uint16_t msg_size;  // total size
    vos_uint32_t msg_id;    // refer to 'EnMsgID'
    vos_uint32_t msg_type;	// refer to 'EnMsgType'
    vos_uint32_t msg_seq;
}MsgHeader;

typedef struct HostAddr
{
    char ip[MAX_IP_LEN+1];
    vos_uint16_t port;
}HostAddr;

typedef struct HistoryRecordBlock
{
    vos_uint32_t begin_time;
    vos_uint32_t end_time;
}HistoryRecordBlock;

typedef struct ChannelStatus
{
    vos_uint8_t channel_idx;  // 1~255
    vos_uint8_t status;       // 0:offline 1:online
}ChannelStatus;

// 音频编解码信息定义
typedef struct AudioCodecInfo
{
    vos_uint8_t codec_fmt;    // 音频格式, 参考 "EnAudioEncodeType"
    vos_uint8_t channel;      // 音频通道, 参考 "EnAudioChannelType"
    vos_uint8_t sample;       // 音频采样率, 参考 "EnAudioSampleRateType"
    vos_uint8_t bitwidth;     // 位宽, 参考 "EnAudioBitwidthType"
    vos_uint8_t sepc_size;    // 音频详细信息长度
    vos_uint8_t sepc_data[8]; // 音频详细信息,具体参考文档
}AudioCodecInfo;

// 视频编解码信息定义
typedef struct VideoCodecInfo
{
    vos_uint8_t codec_fmt;    // 视频格式, 参考 "EnVideoEncodeType"
}VideoCodecInfo;

// 设备属性定义
typedef struct DevAttribute
{
    vos_uint8_t has_microphone;   // 是否有拾音器: 0 没有, 1 有
    vos_uint8_t has_hard_disk;    // 是否有硬盘:   0 没有, 1 有
    vos_uint8_t can_recv_audio;   // 可以接受音频: 0 不支持, 1 支持
}DevAttribute;

// 通道的码流信息定义
typedef struct DevStreamInfo
{
    vos_uint8_t stream_id;            // 码流ID, 参考"EnStreamType"
    vos_uint32_t video_height;        // 视频高度
    vos_uint32_t video_width;         // 视频宽度
    VideoCodecInfo video_codec; // 视频编码信息
}DevStreamInfo;

// 设备的通道信息定义
typedef struct DevChannelInfo
{
    vos_uint16_t channel_id;        // 通道ID, 起始ID从1开始
    vos_uint16_t channel_type;      // 通道类型
    vos_uint8_t channel_status;     // 通道状态, 参考"EnChannelStatus"
    vos_uint8_t has_ptz;            // 是否有云台: 0 没有, 1 有
    vos_uint8_t stream_num;         // 码流数量
    DevStreamInfo stream_list[3];   // 码流信息列表
    AudioCodecInfo audio_codec;     // 音频编码信息
}DevChannelInfo;

#pragma pack()

//#define __ADD_CHK(d, len, max)  d += len;if(d>max){break;}d=d

EXT int Pack_MsgHeader(OUT void* buf, IN vos_size_t buf_len, IN MsgHeader* msg_head);
EXT int Unpack_MsgHeader(IN void* head_buf, IN vos_size_t head_len, OUT MsgHeader* msg_head);

EXT int Pack_DevStreamInfo(OUT void* buf, IN vos_size_t buf_len, IN DevStreamInfo* stream_info);
EXT int Pack_DevChannelInfo(OUT void* buf, IN vos_size_t buf_len, IN DevChannelInfo* channel_info);

#endif
