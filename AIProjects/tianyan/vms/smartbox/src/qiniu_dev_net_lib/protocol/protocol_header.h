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
    EN_ERR_SERVICE_UNAVAILABLE,     // ���񲻿���
    EN_ERR_MALLOC_FAIL,             // ���񲻿���
    EN_ERR_MSG_PARSER_FAIL,         // ��Ϣ��������      
    EN_ERR_TOKEN_CHECK_FAIL,        // tokenУ�����
    EN_ERR_DEVICE_OFFLINE,          // �豸������
    EN_ERR_CHANNEL_OFFLINE,         // �豸������
    EN_ERR_ENDPOINT_UNKWON,         // �ڵ�����δ֪
    EN_ERR_NOT_SUPPORT,             // ��֧�ֵ�ҵ���

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

// �豸���Ͷ���
enum EnDeviceType
{
    DEV_TYPE_IPC     = 0,
    DEV_TYPE_NVR     = 1,
    DEV_TYPE_DVR     = 2,
    DEV_TYPE_QNSMB   = 3     /* ��ţ���ܺ��ӣ�QiNiu SmartBox��*/
};

/* ֡���Ͷ��� */
enum EnFrameType
{
    FRAME_TYPE_I    = 0,    // I֡
    FRAME_TYPE_P    = 1,    // P֡
    FRAME_TYPE_AU   = 2     // ��Ƶ֡
};

// ��Ƶ�������Ͷ���
enum EnVideoEncodeType
{
    VIDEO_H264  = 0,
    VIDEO_H265  = 1,
};

// ��Ƶ�������Ͷ���
enum EnAudioEncodeType
{
    AUDIO_AAC               = 0,
    AUDIO_G711_A            = 1,
    AUDIO_G711_U            = 2,
    AUDIO_MP3               = 3,
};

// ��Ƶͨ�����Ͷ���
enum EnAudioChannelType
{
    AUDIO_CH_MONO           = 0,    // ������
    AUDIO_CH_STEREO         = 1,    // ��������
};
#define  IsAudioCH_Mono(x)      ( (x)==AUDIO_CH_MONO )      // �Ƿ��ǵ�����
#define  IsAudioCH_Stereo(x)    ( (x)==AUDIO_CH_STEREO )    // �Ƿ�����������

// ��Ƶλ�����Ͷ���
enum EnAudioBitwidthType
{
    AUDIO_BW_8BIT           = 0,    //λ��8bit
    AUDIO_BW_16BIT          = 1,    //λ��16bit
};
#define  IsAudioBW_8Bit(x)  ( (x)==AUDIO_BW_8BIT )      //�Ƿ���8bitλ��
#define  IsAudioBW_16Bit(x) ( (x)==AUDIO_CH_STEREO )    //�Ƿ���16bitλ��

// ��Ƶ���������Ͷ���
enum EnAudioSampleRateType
{
    AUDIO_SR_8_KHZ       = 0,    // �����ʣ�8khz
    AUDIO_SR_11_025_KHZ  = 1,    // �����ʣ�11.025khz
    AUDIO_SR_12_KHZ      = 2,    // �����ʣ�12khz
    AUDIO_SR_16_KHZ      = 3,    // �����ʣ�16khz
    AUDIO_SR_22_05_KHZ   = 4,    // �����ʣ�22.05khz
    AUDIO_SR_24_KHZ      = 5,    // �����ʣ�24khz
    AUDIO_SR_32_KHZ      = 6,    // �����ʣ�32khz
    AUDIO_SR_44_1_KHZ    = 7,    // �����ʣ�44.1khz
    AUDIO_SR_48_KHZ      = 8,    // �����ʣ�48khz
    AUDIO_SR_64_KHZ      = 9,    // �����ʣ�64khz
    AUDIO_SR_88_2_KHZ    = 10,   // �����ʣ�88.2khz
    AUDIO_SR_96_KHZ      = 11    // �����ʣ�96khz
};

// �������Ͷ���
enum EnStreamType
{
    STREAM_TYPE_MAIN        = 0,    // ������
    STREAM_TYPE_SUB01       = 1,    // ������1
    STREAM_TYPE_SUB02       = 2,    // ������2
};

/* ͨ��״̬���� */
enum EnChannelStatus
{
    CHANNEL_STS_OFFLINE     = 0,    // ͨ��״̬����
    CHANNEL_STS_ONLINE      = 1,    // ͨ��״̬����
};

// �澯���Ͷ���
enum EnAlarmType
{
    //ƽ̨�澯���Ͷ���
    ALARM_TYPE_DEVICE_ONLINE    = 0x00000001,   // �豸����
    ALARM_TYPE_DEVICE_OFFLINE   = 0x00000002,   // �豸����

    //�豸�˸澯���Ͷ���
    ALARM_TYPE_MOTION_DETECT    = 0x00010001,   // �ƶ����
    ALARM_TYPE_VIDEO_LOST       = 0x00010002,   // ��Ƶ��ʧ
    ALARM_TYPE_VIDEO_SHELTER    = 0x00010003,   // ��Ƶ�ڵ�	
    ALARM_TYPE_DISC             = 0x00010004,   // ���̸澯
    ALARM_TYPE_RECORD           = 0x00010005,   // ¼��澯
};

// �澯״̬����
enum EnAlarmStatus
{
    ALARM_STS_CLEAN      = 0,   // �澯��ȥ
    ALARM_STS_ACTIVE     = 1,   // �澯����
    ALARM_STS_KEEP       = 2,   // �澯����
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

// ��Ƶ�������Ϣ����
typedef struct AudioCodecInfo
{
    vos_uint8_t codec_fmt;    // ��Ƶ��ʽ, �ο� "EnAudioEncodeType"
    vos_uint8_t channel;      // ��Ƶͨ��, �ο� "EnAudioChannelType"
    vos_uint8_t sample;       // ��Ƶ������, �ο� "EnAudioSampleRateType"
    vos_uint8_t bitwidth;     // λ��, �ο� "EnAudioBitwidthType"
    vos_uint8_t sepc_size;    // ��Ƶ��ϸ��Ϣ����
    vos_uint8_t sepc_data[8]; // ��Ƶ��ϸ��Ϣ,����ο��ĵ�
}AudioCodecInfo;

// ��Ƶ�������Ϣ����
typedef struct VideoCodecInfo
{
    vos_uint8_t codec_fmt;    // ��Ƶ��ʽ, �ο� "EnVideoEncodeType"
}VideoCodecInfo;

// �豸���Զ���
typedef struct DevAttribute
{
    vos_uint8_t has_microphone;   // �Ƿ���ʰ����: 0 û��, 1 ��
    vos_uint8_t has_hard_disk;    // �Ƿ���Ӳ��:   0 û��, 1 ��
    vos_uint8_t can_recv_audio;   // ���Խ�����Ƶ: 0 ��֧��, 1 ֧��
}DevAttribute;

// ͨ����������Ϣ����
typedef struct DevStreamInfo
{
    vos_uint8_t stream_id;            // ����ID, �ο�"EnStreamType"
    vos_uint32_t video_height;        // ��Ƶ�߶�
    vos_uint32_t video_width;         // ��Ƶ���
    VideoCodecInfo video_codec; // ��Ƶ������Ϣ
}DevStreamInfo;

// �豸��ͨ����Ϣ����
typedef struct DevChannelInfo
{
    vos_uint16_t channel_id;        // ͨ��ID, ��ʼID��1��ʼ
    vos_uint16_t channel_type;      // ͨ������
    vos_uint8_t channel_status;     // ͨ��״̬, �ο�"EnChannelStatus"
    vos_uint8_t has_ptz;            // �Ƿ�����̨: 0 û��, 1 ��
    vos_uint8_t stream_num;         // ��������
    DevStreamInfo stream_list[3];   // ������Ϣ�б�
    AudioCodecInfo audio_codec;     // ��Ƶ������Ϣ
}DevChannelInfo;

#pragma pack()

//#define __ADD_CHK(d, len, max)  d += len;if(d>max){break;}d=d

EXT int Pack_MsgHeader(OUT void* buf, IN vos_size_t buf_len, IN MsgHeader* msg_head);
EXT int Unpack_MsgHeader(IN void* head_buf, IN vos_size_t head_len, OUT MsgHeader* msg_head);

EXT int Pack_DevStreamInfo(OUT void* buf, IN vos_size_t buf_len, IN DevStreamInfo* stream_info);
EXT int Pack_DevChannelInfo(OUT void* buf, IN vos_size_t buf_len, IN DevChannelInfo* channel_info);

#endif
