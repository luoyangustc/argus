
#ifndef __PROTOCOL_DEVICE_H__
#define __PROTOCOL_DEVICE_H__

#include "vos_types.h"
#include "protocol_header.h"

#undef  EXT
#ifndef __PROTOCOL_DEVICE_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

#define MAX_MEDIA_SESSION_ID_LEN    (64)
#define MAX_DEV_ID_LEN              (64)
#define MAX_DEV_VERSION_LEN         (64)
#define MAX_DEV_CHANNEL_NUM         (256)
#define MAX_PIC_FMT_LEN             (20)

enum DeviceCMDType
{
    DEVICE_CMD_PTZ          = 0x0001, // 云台控制命令
    DEVICE_CMD_REBOOT       = 0x0002, // 设备重启命令
    DEVICE_CMD_RESET        = 0x0003, // 设备重置命令
    DEVICE_CMD_MGR_UPDATE   = 0x0004, // 设备管理状态更新指令
};

enum MediaDirectType
{
    MEDIA_DIR_SEND_ONLY  = 0x01,
    MEDIA_DIR_RECV_ONLY  = 0x02,
    MEDIA_DIR_SEND_RECV  = 0x03
};

#define  MediaFrameMask 0x7F
#define  IsExpireFrame(x) ( ( (x)&~MediaFrameMask)!=0 )

enum MediaCMDType
{
    MEDIA_CMD_VIDEO_OPEN    = 0x0001,
    MEDIA_CMD_VIDEO_CLOSE   = 0x0002,
    MEDIA_CMD_AUDIO_OPEN    = 0x0003,
    MEDIA_CMD_AUDIO_CLOSE   = 0x0004
};

#pragma pack(1)

typedef struct __token_t 
{ 
	vos_uint16_t token_bin_length; 
	vos_uint8_t token_bin[256]; 
}token_t;

typedef struct DeviceMediaSessionStatus
{
    char            session_id[MAX_MEDIA_SESSION_ID_LEN+1];
    vos_uint16_t    session_type;    // refer to 'MediaSessionType'
    vos_uint8_t     session_media;    // 1:video, 2:audio, 3:all(video+audio)
    vos_uint8_t     session_status;   // 0x00:close 01: building  0x02:running
    char            device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t    channel_id;
    vos_uint8_t     stream_id;
    HostAddr        stream_addr;
}DeviceMediaSessionStatus;

struct DeviceLoginReq
{
    vos_uint32_t mask;

    //0x01
    char device_id[MAX_DEV_ID_LEN+1];
    char version[MAX_DEV_VERSION_LEN+1];
    vos_uint8_t dev_type;     // 1:dvr, 2:nvr, 3:ipc 	
    vos_uint16_t channel_num;
    DevChannelInfo* channels;
    token_t token;

    //0x02
    HostAddr private_addr;
};
struct DeviceLoginResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0X01
    HostAddr public_addr;
};
typedef struct DeviceLoginReq DeviceLoginReq;
typedef struct DeviceLoginResp DeviceLoginResp;


struct DeviceAbilityReportReq
{
    vos_uint32_t mask;

    //0x01
    vos_uint8_t media_trans_type;           // 1-TCP, 2-UDP, 3-ALL(TCP、UDP)
    vos_uint8_t max_live_streams_per_ch;
    vos_uint8_t max_playback_streams_per_ch;
    vos_uint8_t max_playback_streams;

    //0x02 
    vos_uint32_t disc_size;
    vos_uint32_t disc_free_size;
};
struct DeviceAbilityReportResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;
};
typedef struct DeviceAbilityReportReq DeviceAbilityReportReq;
typedef struct DeviceAbilityReportResp DeviceAbilityReportResp;

struct DeviceStatusReportReq
{
    vos_uint32_t mask;

    //0x01
    vos_uint16_t channel_num;
    DevChannelInfo* channels;

    //0x02
    vos_uint16_t media_session_num;
    DeviceMediaSessionStatus* media_sessions;

    //0x04
    vos_uint8_t sdcard_status;  // refer to 'Sdcard_Status'
};
struct DeviceStatusReportResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    vos_uint16_t expected_cycle;
};
typedef struct DeviceStatusReportReq DeviceStatusReportReq;
typedef struct DeviceStatusReportResp DeviceStatusReportResp;

struct DeviceAlarmReportReq
{
    vos_uint32_t mask;

    //0x01
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
    vos_uint32_t alarm_type;
    vos_uint8_t alarm_status;

    //0x02
    vos_uint32_t alarm_data_size;
    vos_uint8_t* alarm_datas;
};
struct DeviceAlarmReportResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;
};
typedef struct DeviceAlarmReportReq DeviceAlarmReportReq;
typedef struct DeviceAlarmReportResp DeviceAlarmReportResp;

struct DeviceCtrlReq
{
    vos_uint32_t mask;

    //0x01
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
    vos_uint16_t cmd_type;    //refer to 'DeviceCMDType'

    //0x02
    uint16 cmd_data_size;	    // 指令数据大小
    vos_uint8_t cmd_datas[512]; // 指令数据
};
struct DeviceCtrlResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;
};
typedef struct DeviceCtrlReq    DeviceCtrlReq;
typedef struct DeviceCtrlResp   DeviceCtrlResp;

struct DeviceMediaOpenReq
{
    vos_uint32_t mask;

    //0x01
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
    vos_uint8_t stream_id;        //refer to 'RateType'
    vos_uint16_t session_type;    // refer to 'MediaSessionType'
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
    vos_uint8_t session_media;       // 0x01:Video 0x02:Audio, 0x03:all

    //0x02
    VideoCodecInfo video_codec;

    //0x04
    AudioCodecInfo audio_codec;

    //0x08
    vos_uint8_t transport_type;   // 1:ES_OVER_TCP 2:ES_OVER_UDP, default:1

    //0x10
    vos_uint32_t begin_time;
    vos_uint32_t end_time;

    //0x20
    token_t stream_token;
    vos_uint16_t stream_num;
    HostAddr stream_servers[10];
};

struct DeviceMediaOpenResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
    vos_uint8_t stream_id;        //refer to 'RateType'
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
    HostAddr stream_server;
};
typedef struct DeviceMediaOpenReq DeviceMediaOpenReq;
typedef struct DeviceMediaOpenResp DeviceMediaOpenResp;


struct DeviceMediaCloseReq
{
    vos_uint32_t mask;

    //0x01 
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
    vos_uint8_t stream_id;        //refer to 'RateType'
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
struct DeviceMediaCloseResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;
};
typedef struct DeviceMediaCloseReq DeviceMediaCloseReq;
typedef struct DeviceMediaCloseResp DeviceMediaCloseResp;

struct DeviceSnapReq
{
    vos_uint32_t mask;

    //0x01 
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
};
struct DeviceSnapResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01 
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
    char pic_fmt[MAX_PIC_FMT_LEN+1];
    vos_uint32_t pic_size;

    //0x02
    vos_uint32_t offset;
    vos_uint32_t data_size;
    vos_uint8_t datas[MAX_MSG_BUFF_SIZE];
};
typedef struct DeviceSnapReq    DeviceSnapReq;
typedef struct DeviceSnapResp   DeviceSnapResp;

struct StreamMediaConnectReq
{
    vos_uint32_t mask;

    //0x01
    vos_uint16_t session_type;    // refer to 'MediaSessionType'
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
    vos_uint8_t session_media;	// 0x01:Video 0x02:Audio, 0x03:all
    char endpoint_name[MAX_DEV_ID_LEN+1];
    vos_uint16_t endpoint_type;   // 1:Device 2:CU 3:StreamServ(Relay) refer to 'EndPointType'
    char device_id[MAX_DEV_ID_LEN+1];
    vos_uint16_t channel_id;
    vos_uint8_t stream_id;        // refer to 'EnStreamType'
    token_t token;

    //0x02
    vos_uint8_t video_direct;     // 0x01:sendonly, 0x02:recvonly, 0x03:sendrecv        
    VideoCodecInfo video_codec;

    //0x04
    vos_uint8_t audio_direct;     // 0x01:sendonly, 0x02:recvonly, 0x03:sendrecv
    AudioCodecInfo audio_codec;

    //0x08
    vos_uint32_t begin_time;
    vos_uint32_t end_time;
};
struct StreamMediaConnectResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaConnectReq StreamMediaConnectReq;
typedef struct StreamMediaConnectResp StreamMediaConnectResp;

struct StreamMediaDisconnectReq
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
struct StreamMediaDisconnectResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaDisconnectReq StreamMediaDisconnectReq;
typedef struct StreamMediaDisconnectResp StreamMediaDisconnectResp;

struct StreamMediaPlayReq
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];        

    //0x02
    vos_uint32_t begin_time;
    vos_uint32_t end_time;

    //0x04
    vos_uint8_t speed;
};
struct StreamMediaPlayResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaPlayReq StreamMediaPlayReq;
typedef struct StreamMediaPlayResp StreamMediaPlayResp;

struct StreamMediaPauseReq
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
struct StreamMediaPauseResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaPauseReq StreamMediaPauseReq;
typedef struct StreamMediaPauseResp StreamMediaPauseResp;

struct StreamMediaCmdReq
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
    vos_uint8_t cmd_type;     // refer to 'MediaCMDType'
        
    //0x02
    vos_uint16_t param_data_size;
    char param_datas[512];
};
struct StreamMediaCmdResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaCmdReq StreamMediaCmdReq;
typedef struct StreamMediaCmdResp StreamMediaCmdResp;

struct StreamMediaStatusReq
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];

    //0x02
    vos_uint8_t video_status;     // 1:on 2:off

    //0x04
    vos_uint8_t audio_status;     // 1:on 2:off
};
struct StreamMediaStatusResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaStatusReq StreamMediaStatusReq;
typedef struct StreamMediaStatusResp StreamMediaStatusResp;

struct StreamMediaCloseReq
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
struct StreamMediaCloseResp
{
    vos_uint32_t mask;
    vos_int32_t resp_code;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaCloseReq StreamMediaCloseReq;
typedef struct StreamMediaCloseResp StreamMediaCloseResp;

struct StreamMediaFrameNotify
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];

    //0x02
    vos_uint8_t frame_type;
    vos_uint32_t frame_av_seq;
    vos_uint32_t frame_seq;
    vos_uint32_t frame_base_time;
    vos_uint32_t frame_ts;
    vos_uint32_t frame_size;

    //0X04
    vos_uint32_t crc32_hash;

    //0x08
    vos_uint32_t offset;
    vos_uint32_t data_size;
    vos_uint8_t datas[MAX_MSG_BUFF_SIZE];
};
typedef struct StreamMediaFrameNotify StreamMediaFrameNotify;

struct StreamMediaEosNotify
{
    vos_uint32_t mask;

    //0x01
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
};
typedef struct StreamMediaEosNotify StreamMediaEosNotify;

#pragma pack()

//@ pack message
//@ return： success(>0 package len), error(<0) 
EXT int Pack_MsgDeviceLoginReq(OUT void* buf, IN vos_size_t buf_len, IN DeviceLoginReq* msg);
EXT int Pack_MsgDeviceAbilityReportReq(OUT char* buf, IN vos_size_t buf_len, IN DeviceAbilityReportReq* msg);
EXT int Pack_MsgDeviceStatusReportReq(OUT char* buf, IN vos_size_t buf_len, IN DeviceStatusReportReq* msg);
EXT int Pack_MsgDeviceAlarmReportReq(OUT char* buf, IN vos_size_t buf_len, IN DeviceAlarmReportReq* msg);
EXT int Pack_MsgDeviceMediaOpenResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceMediaOpenResp* msg);
EXT int Pack_MsgDeviceMediaCloseResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceMediaCloseResp* msg);
EXT int Pack_MsgDeviceSnapResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceSnapResp* msg);
EXT int Pack_MsgDeviceCtrlResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceCtrlResp* msg);

EXT int Pack_MsgStreamMediaConnectReq(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaConnectReq* msg);
EXT int Pack_MsgStreamMediaDisconnectReq(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaDisconnectReq* msg);
EXT int Pack_MsgStreamMediaStatusReq(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaStatusReq* msg);
EXT int Pack_MsgStreamMediaPlayResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaPlayResp* msg);
EXT int Pack_MsgStreamMediaPauseResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaPauseResp* msg);
EXT int Pack_MsgStreamMediaCmdResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaCmdResp* msg);
EXT int Pack_MsgStreamMediaCloseResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaCloseResp* msg);
EXT int Pack_MsgStreamMediaFrameNotify(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaFrameNotify* msg);
EXT int Pack_MsgStreamMediaEosNotify(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaEosNotify* msg);

//@ unpack message
//@ return： success(0), error(<0) 
EXT int Unpack_MsgDeviceLoginResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceLoginResp* msg);
EXT int Unpack_MsgDeviceAbilityReportResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceAbilityReportResp* msg);
EXT int Unpack_MsgDeviceStatusReportResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceStatusReportResp* msg);
EXT int Unpack_MsgDeviceAlarmReportResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceAlarmReportResp* msg);
EXT int Unpack_MsgDeviceMediaOpenReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceMediaOpenReq* msg);
EXT int Unpack_MsgDeviceMediaCloseReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceMediaCloseReq* msg);
EXT int Unpack_MsgDeviceSnapReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceSnapReq* msg);
EXT int Unpack_MsgDeviceCtrlReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceCtrlReq* msg);

EXT int Unpack_MsgStreamMediaConnectResp(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaConnectResp* msg);
EXT int Unpack_MsgStreamMediaDisconnectResp(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaDisconnectResp* msg);
EXT int Unpack_MsgStreamMediaStatusResp(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaStatusResp* msg);
EXT int Unpack_MsgStreamMediaPlayReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaPlayReq* msg);
EXT int Unpack_MsgStreamMediaPauseReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaPauseReq* msg);
EXT int Unpack_MsgStreamMediaCmdReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaCmdReq* msg);
EXT int Unpack_MsgStreamMediaCloseReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaCloseReq* msg);

VOS_END_DECL

#endif
