#ifndef __protocol_HEADER_H__
#define __protocol_HEADER_H__
#include <vector>
#include <string>
using namespace std;
#include "typedefine.h"
#include "datastream.h"

namespace protocol{
 
#pragma pack (push, 1)
 
    #define MAX_MSG_BUFF_SIZE	        (8*1024)
    #define MAX_TCP_PACKET_SIZE		    (64*1024)
    #define MAX_MEDIA_SESSION_ID_LEN    20

    enum EnMsgType
    {
        MSG_TYPE_REQ    = 0x00000001,   //request msg
        MSG_TYPE_RESP   = 0x00000002,   //response msg
        MSG_TYPE_NOTIFY = 0x00000003,   //notify msg
    };
    
    enum EnMsgID
    {
        //Exchange key
        MSG_ID_EXCHANGE_KEY             = 0x00000001,

        //IPC TUNNEL
        MSG_ID_TUNNEL                   = 0x00000002,

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
        MSG_ID_STS_STREAM_STATUS_REPORT     = 0x06000004,
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
        EN_DEV_ERR_START                = -600,
        EN_DEV_ERR_ALREADY_LOGIN,
        EN_DEV_ERR_GET_STREAM_SERV_FAIL,
        EN_DEV_ERR_CONNECT_STREAM_REPEAT,
        EN_DEV_ERR_CREATE_STREAM_FAIL,
        EN_DEV_ERR_STREAM_DISCONNECT,
        EN_DEV_ERR_STREAM_DIFFERENT,

        //for client
        EN_CU_ERR_GET_STREAM_SERV_FAIL     = -300,
        EN_CU_ERR_CONNECT_STREAM_REPEAT,
        EN_CU_ERR_CREATE_STREAM_FAIL,
        EN_CU_ERR_STREAM_DISCONNECT,
    };
    
    // 节点类型定义
    enum EndPointType
    {
        EP_UNKNOWN      = 0x00,     // 未知
        EP_DEV   	    = 0x01,     // 设备端
        EP_CU           = 0x02,     // 客户端
        EP_STREAM       = 0x03,     // 流服务器
        EP_SMS          = 0x04,     // 会话服务器
    };
    
    // 设备类型定义
    enum EnDeviceType
    {
        DEV_TYPE_IPC     = 0,
        DEV_TYPE_NVR     = 1,
        DEV_TYPE_DVR     = 2,
        DEV_TYPE_QNSMB   = 3,     /* 七牛智能盒子（QiNiu SmartBox）*/
        DEV_TYPE_MAX
    };

    /* 帧类型定义 */
    enum EnFrameType
    {
        FRAME_TYPE_I            = 0,    // I帧
        FRAME_TYPE_P            = 1,    // P帧
        FRAME_TYPE_AU           = 2     // 音频帧
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

    
    // 媒体会话类型定义
    enum EnMediaSessionType
    {
        MEDIA_SESSION_TYPE_LIVE                 = 0x0001,    // 实时浏览
        MEDIA_SESSION_TYPE_PU_PLAYBACK          = 0x0002,    // 前端(NVR/TF卡)录像回放
        MEDIA_SESSION_TYPE_PU_DOWNLOAD          = 0x0003,    // 前端(NVR/TF卡)录像下载
        MEDIA_SESSION_TYPE_DIRECT_LIVE          = 0x8001,    // 直连实时浏览
        MEDIA_SESSION_TYPE_DIRECT_PU_PLAYBACK   = 0x8002,    // 直连前端(NVR/TF卡)录像回放
        MEDIA_SESSION_TYPE_DIRECT_PU_DOWNLOAD   = 0x8003,    // 直连前端(NVR/TF卡)录像下载
    };

    struct MsgHeader
    {
        uint16 msg_size;	// total size
        uint32 msg_id;      // refer to 'EnMsgID'
        uint32 msg_type;	// refer to 'EnMsgType'
        uint32 msg_seq;     // 序号，请求和响应要一致
    };

    typedef struct MsgHeader MSG_HEADER;

    // 主机信息定义
    struct HostAddr
    {
        string ip;
        uint16 port;
    };
    
    // 历史录像块定义
    struct HistoryRecordBlock
    {
        uint32 begin_time;
        uint32 end_time;
    };

    //带宽信息
    struct SBandwidthInfo
    {
        uint16 upload_bw;       //上行带宽(kbps)
        uint16 download_bw;     //下行带宽(kbps)
    }; 

    //时间戳信息
    struct STimeVal
    {
        uint64  tv_sec;
        uint64  tv_usec;

        bool operator<(const STimeVal& right)const
        {
            if(tv_sec < right.tv_sec)
            {
                return true;
            }
            else if( (tv_sec == right.tv_sec) && ( tv_usec < right.tv_usec) )
            {
                return true;
            }

            return false;	
        }

        bool operator==(const STimeVal& right)const
        {
            if( (tv_sec == right.tv_sec) && ( tv_usec == right.tv_usec) )
            {
                return true;
            }

            return false;
        }
        bool operator!=(const STimeVal& right)const
        {
            return !(*this == right);
        }

        bool operator<=(const STimeVal& right)const
        {
            return (*this < right) || (*this == right);
        }

        bool operator>(const STimeVal& right)const
        {
            return (right < *this);
        }

        bool operator>=(const STimeVal& right)const
        {
            return (right < *this) || (*this == right);
        }
    };
    
    // 通道状态定义
    struct ChannelStatus
    {
        uint16 channel_id;  // 1~255
        uint8 status;       // 0:offline 1:online
    };
    
    // 音频编解码信息定义
    struct AudioCodecInfo
    {
        uint8 codec_fmt;            // 音频格式, 参考 "EnAudioEncodeType"
        uint8 channel;              // 音频通道, 参考 "EnAudioChannelType"
        uint8 sample;               // 音频采样率, 参考 "EnAudioSampleRateType"
        uint8 bitwidth;             // 位宽, 参考 "EnAudioBitwidthType"
        uint8 sepc_size;            // 音频详细信息长度
        vector<uint8> sepc_data;    // 音频详细信息,具体参考文档
    };

    // 视频编解码信息定义
    struct VideoCodecInfo
    {
        uint8 codec_fmt;    // 视频格式, 参考 "EnVideoEncodeType"
    };

    // 设备属性定义
    struct DevAttribute
    {
        uint8 has_microphone;   // 是否有拾音器: 0 没有, 1 有
        uint8 has_hard_disk;    // 是否有硬盘:   0 没有, 1 有
        uint8 can_recv_audio;   // 可以接受音频: 0 不支持, 1 支持
    };

    // 设备OEM信息定义
    struct DevOEMInfo
    {
        uint32 oem_id;       // OEM ID
        string oem_name;    // 厂商OEM名称,具体根据不同厂商由平台统一提供 
        string mac;         // MAC 地址
        string sn;          // 序列号		    
        string model;       // 设备型号
        string factory;     // 厂商名称(eg. 海康威视、大华等)
    };

    // 通道的码流信息定义
    struct DevStreamInfo
    {
        uint8 stream_id;            // 码流ID, 参考"EnStreamType"
        uint32 video_height;        // 视频高度
        uint32 video_width;         // 视频宽度
        VideoCodecInfo video_codec; // 视频编码信息
    };

    // 设备的通道信息定义
    struct DevChannelInfo
    {
        uint16 channel_id;                  // 通道ID, 起始ID从1开始
        uint16 channel_type;                // 通道类型
        uint8 channel_status;               // 通道状态, 参考"EnChannelStatus"
        uint8 has_ptz;                      // 是否有云台: 0 没有, 1 有
        uint8 stream_num;                   // 码流数量
        vector<DevStreamInfo> stream_list;  // 码流信息列表
        AudioCodecInfo audio_codec;         // 音频编码信息
    };

    //进程间通信隧道
    struct MsgTunnelReq
    {
        uint32 mask;
        //0x01
        uint32 tunnel_type;
        //0x02
        uint32 req_data_size;
        vector<uint8> req_datas;
    };
    
    struct MsgTunnelResp
    {
        uint32 mask;
        int32 resp_code;

        //0x01
        uint32 resp_data_size;
        vector<uint8> resp_datas;
    };

    //序列化
    CDataStream& operator<<(CDataStream& ds, MsgHeader& msg);
    CDataStream& operator>>(CDataStream& ds, MsgHeader& msg);
    
    CDataStream& operator<<( CDataStream& ds,token_t & token );
    CDataStream& operator>>( CDataStream& ds, token_t& token );

    CDataStream& operator<<(CDataStream& ds, HostAddr& hi);
    CDataStream& operator>>(CDataStream& ds, HostAddr& hi);

    CDataStream& operator<<(CDataStream& ds, HistoryRecordBlock& blk);
    CDataStream& operator>>(CDataStream& ds, HistoryRecordBlock& blk);

    CDataStream& operator<<(CDataStream& ds, SBandwidthInfo& bw);
    CDataStream& operator>>(CDataStream& ds, SBandwidthInfo& bw);

    CDataStream& operator<<(CDataStream& ds, STimeVal& tv);
    CDataStream& operator>>(CDataStream& ds, STimeVal& tv);

    CDataStream& operator<<(CDataStream& ds, ChannelStatus& status);
    CDataStream& operator>>(CDataStream& ds, ChannelStatus& status);

    CDataStream& operator<<(CDataStream& ds, AudioCodecInfo& data);
    CDataStream& operator>>(CDataStream& ds, AudioCodecInfo& data);

    CDataStream& operator<<(CDataStream& ds, VideoCodecInfo& data);
    CDataStream& operator>>(CDataStream& ds, VideoCodecInfo& data);

    CDataStream& operator<<(CDataStream& ds, DevAttribute& data);
    CDataStream& operator>>(CDataStream& ds, DevAttribute& data);

    CDataStream& operator<<(CDataStream& ds, DevOEMInfo& data);
    CDataStream& operator>>(CDataStream& ds, DevOEMInfo& data);
    
    CDataStream& operator<<(CDataStream& ds, DevStreamInfo& data);
    CDataStream& operator>>(CDataStream& ds, DevStreamInfo& data);

    CDataStream& operator<<(CDataStream& ds, DevChannelInfo& data);
    CDataStream& operator>>(CDataStream& ds, DevChannelInfo& data);

    CDataStream& operator<<(CDataStream& ds, MsgTunnelReq& req);
    CDataStream& operator>>(CDataStream& ds, MsgTunnelReq& req);

    CDataStream& operator<<(CDataStream& ds, MsgTunnelResp& resp);
    CDataStream& operator>>(CDataStream& ds, MsgTunnelResp& resp);

#pragma pack (pop)
};

#endif //__protocol_HEADER_H__

