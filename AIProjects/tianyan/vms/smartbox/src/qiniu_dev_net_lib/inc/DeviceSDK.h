#ifndef __DEVICE_SDK_H__
#define __DEVICE_SDK_H__

typedef char                sdk_int8;
typedef short 				sdk_int16;
typedef int   				sdk_int32;
typedef unsigned char 		sdk_uint8;
typedef unsigned short 		sdk_uint16;
typedef unsigned int  		sdk_uint32;
typedef unsigned long long 	sdk_uint64;

#ifdef WIN32_EXPORT
#define SDK_API __declspec(dllexport)
#elif WIN32_IMPORT
#define SDK_API __declspec(dllimport)
#else
#define SDK_API 
#endif

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

    /* �豸���Ͷ��� */
    typedef enum en_device_type
    {
        EN_DEV_TYPE_IPC     = 0,
        EN_DEV_TYPE_NVR     = 1,
        EN_DEV_TYPE_DVR     = 2,
        EN_DEV_TYPE_QNSMB   = 3     /* ��ţ���ܺ��ӣ�QiNiu SmartBox��*/
    }EN_DEV_TYPE;

    /* �������Ͷ��� */
    typedef enum en_stream_type
    {
        EN_STREAM_TYPE_MAIN     = 0,    /* ������  */
        EN_STREAM_TYPE_SUB01    = 1     /* ������1 */
    }EN_STREAM_TYPE;

    /* ͨ��״̬���� */
    typedef enum en_channel_status
    {
        EN_CH_STS_OFFLINE   = 0,    /* ͨ��״̬����  */
        EN_CH_STS_ONLINE    = 1     /* ͨ��״̬����  */
    }EN_CH_STATUS;
     
    /* ֡���Ͷ��� */
    typedef enum en_frame_type
    {
        EN_FRM_TYPE_I   = 0,    /* I֡    */
        EN_FRM_TYPE_P   = 1,    /* P֡    */
        EN_FRM_TYPE_AU  = 2     /* ��Ƶ֡ */
    }EN_FRAME_TYPE;
    
    /* ͼƬ��ʽ���Ͷ��� */
    typedef enum en_pic_fmt_type
    {
        EN_PIC_FMT_BMP  = 0,
        EN_PIC_FMT_JPEG = 1
    }EN_PIC_FMT_TYPE;

    /* ��Ƶ�����ʽ���Ͷ��� */
    typedef enum en_video_fmt_type
    {
        EN_VIDEO_FMT_H264   = 0,
        EN_VIDEO_FMT_H265   = 1
    }EN_VIDEO_FMT_TYPE;

    /* ��Ƶ�����ʽ���Ͷ��� */
    typedef enum en_audio_fmt_type
    {
        EN_AUDIO_FMT_AAC    = 0,
        EN_AUDIO_FMT_G711_A = 1,
        EN_AUDIO_FMT_G711_U = 2,
        EN_AUDIO_FMT_MP3    = 3
    }EN_AUDIO_FMT_TYPE;

    /* ��Ƶͨ�����Ͷ��� */
    typedef enum en_audio_ch_type
    {
        EN_AUDIO_CH_MONO    = 0,    /* ������ */
        EN_AUDIO_CH_STEREO  = 1     /* �������� */
    }EN_AUDIO_CH_TYPE;

    /* ��Ƶλ�����Ͷ��� */
    typedef enum en_audio_bitwidth_type
    {
        EN_AUDIO_BW_8BIT    = 0,    /* λ��8bit */
        EN_AUDIO_BW_16BIT   = 1     /* λ��16bit */
    }EN_AUDIO_BW_TYPE;

    /* ��Ƶ���������Ͷ��� */
    typedef enum en_audio_sr_type
    {
        EN_AUDIO_SR_8_KHZ       = 0,    /* �����ʣ�8khz */
        EN_AUDIO_SR_11_025_KHZ  = 1,    /* �����ʣ�11.025khz */
        EN_AUDIO_SR_12_KHZ      = 2,    /* �����ʣ�12khz */
        EN_AUDIO_SR_16_KHZ      = 3,    /* �����ʣ�16khz */
        EN_AUDIO_SR_22_05_KHZ   = 4,    /* �����ʣ�22.05khz */
        EN_AUDIO_SR_24_KHZ      = 5,    /* �����ʣ�24khz */
        EN_AUDIO_SR_32_KHZ      = 6,    /* �����ʣ�32khz */
        EN_AUDIO_SR_44_1_KHZ    = 7,    /* �����ʣ�44.1khz */
        EN_AUDIO_SR_48_KHZ      = 8,    /* �����ʣ�48khz */
        EN_AUDIO_SR_64_KHZ      = 9,    /* �����ʣ�64khz */
        EN_AUDIO_SR_88_2_KHZ    = 10,   /* �����ʣ�88.2khz */
        EN_AUDIO_SR_96_KHZ      = 11    /* �����ʣ�96khz */
    }EN_AUDIO_SR_TYPE;

    /* �豸�澯���Ͷ��� */
    typedef enum en_alarm_type
    {
        EN_ALARM_MOVE_DETECT    = 0x00010001,   /* �ƶ���� */
        EN_ALARM_VIDEO_LOST     = 0x00010002,   /* ��Ƶ��ʧ */
        EN_ALARM_VIDEO_SHELTER  = 0x00010003,   /* ��Ƶ�ڵ� */	
        EN_ALARM_VIDEO_DISC     = 0x00010004,   /* ���̸澯 */	
        EN_ALARM_VIDEO_RECORD   = 0x00010005    /* ǰ��¼��澯 */
    }EN_ALARM_TYPE;

    /* �澯״̬���� */
    typedef enum en_alarm_status
    {
        EN_ALARM_STS_CLEAN      = 0,   /* �澯��ȥ */
        EN_ALARM_STS_ACTIVE     = 1,   /* �澯���� */
        EN_ALARM_STS_KEEP       = 2    /* �澯���� */
    }EN_ALARM_STATUS;

    /* ����ָ�����Ͷ��� */
    typedef enum en_cmd_type
    {
        EN_CMD_LIVE_OPEN,   /* ��ʵʱ��Ƶ, cmd_agrs-->stream_id(uint8) [0:main_stream, 1:sub_stream] */
        EN_CMD_LIVE_CLOSE,  /* �ر�ʵʱ��Ƶ */
        EN_CMD_SNAP,        /* ץ��ͼƬ */
        EN_CMD_PTZ,         /* ��̨���� */
        EN_CMD_PARAM_GET,   /* ��ȡ�豸���� */
        EN_CMD_PARAM_SET,   /* �����豸���� */
        EN_CMD_MGR_UPDATE   /* �����豸����״̬, cmd_agrs-->mgr_type(uint16) [ 1:add_channel, 2:modify_channel, 3: del_channel ] */
    }EN_CMD_TYPE;

    /* ����ָ��������� */
    typedef struct Dev_Cmd_Param_t
    {
        sdk_uint16  cmd_type;       /* �������ͣ��ο�EN_CMD_TYPE */
        int         channel_index;  /* ͨ�� */
        char        cmd_args[512];	/* �����������������μ��ĵ� */
    }Dev_Cmd_Param_t;

    /* ҵ������sdkָ��Ļص�����ԭ�Ͷ��� */
    typedef void(*Dev_Cmd_Cb_Func)(void* user_data, Dev_Cmd_Param_t* cmd);

    /* ��Ƶ�������Ϣ���� */
    typedef struct Dev_Audio_Codec_Info_t
    {
        sdk_uint8   codec_fmt;      /* ��Ƶ��ʽ, �ο� "EN_AUDIO_FMT_TYPE" */
        sdk_uint8   channel;        /* ��Ƶͨ��, �ο� "EN_AUDIO_CH_TYPE" */
        sdk_uint8   sample;         /* ��Ƶ������, �ο� "EN_AUDIO_SR_TYPE" */
        sdk_uint8   bitwidth;       /* λ��, �ο� "EN_AUDIO_BW_TYPE" */
        sdk_uint8   sepc_size;      /* ��Ƶ��ϸ��Ϣ���� */
        sdk_uint8   sepc_data[8];   /* ��Ƶ��ϸ��Ϣ,����ο��ĵ� */
    }Dev_Audio_Codec_Info_t;

    /* ��Ƶ�������Ϣ���� */
    typedef struct Dev_Video_Codec_Info_t
    {
        sdk_uint8   codec_fmt;    /* ��Ƶ��ʽ, �ο� "EN_VIDEO_FMT_TYPE" */
    }Dev_Video_Codec_Info_t;

    /* �豸���Զ��� */
	typedef struct Dev_Attribute_t
	{
        sdk_uint8   has_microphone;   /* �Ƿ���ʰ����: 0 û��, 1 �� */
		sdk_uint8   has_hard_disk;    /* �Ƿ���Ӳ��:   0 û��, 1 �� */
        sdk_uint8   can_recv_audio;   /* ���Խ�����Ƶ: 0 ��֧��, 1 ֧�� */
	}Dev_Attribute_t;

    /* �豸OEM��Ϣ���� */
    typedef struct Dev_OEM_Info_t
    {
        sdk_int32   OEMID;              /* OEM ID */
        char        OEM_name[2+1];      /* ����OEM����,������ݲ�ͬ������ƽ̨ͳһ�ṩ */
        char        MAC[17 + 1];        /* MAC ��ַ */
        char        SN[16+1];           /* ���к� */			    
        char        Model[64 + 1];      /* �豸�ͺ� */
        char        Factory[128 + 1];   /* ��������(eg. �������ӡ��󻪵�)*/
    }Dev_OEM_Info_t;

    /* ͨ����������Ϣ���� */
    typedef struct Dev_Stream_Info_t
    {
        sdk_uint8               stream_id;      /* ����ID: 0 ������, 1 ������ */
        sdk_uint32              video_height;   /* ��Ƶ�߶� */
        sdk_uint32              video_width;    /* ��Ƶ��� */
        Dev_Video_Codec_Info_t  video_codec;    /* ��Ƶ������Ϣ */
    }Dev_Stream_Info_t;

    /* �豸��ͨ����Ϣ���� */
    typedef struct Dev_Channel_Info_t
    {
        sdk_uint16              channel_index;  /* ͨ��ID����ʼID��1��ʼ */
        sdk_uint16              channel_type;   /* ͨ������, 0: ����ͷ */
        sdk_uint8               channel_status; /* ͨ��״̬, �ο�"EN_CH_STATUS" */
        sdk_uint8               has_ptz;        /* �Ƿ�����̨: 0 û��, 1 �� */
        sdk_uint8               stream_num;     /* �������� */
        Dev_Stream_Info_t*      stream_list;    /* ������Ϣ�б� */
        Dev_Audio_Codec_Info_t  adudo_codec;    /* ��Ƶ������Ϣ */
    }Dev_Channel_Info_t;

    /* ֡��Ϣ���� */
    typedef struct Dev_Stream_Frame_t
    {
        sdk_uint16  channel_index;  /* ͨ���� */
        sdk_uint8  	stream_id;      /* ����ID: 0 ������, 1 ������ */
        sdk_uint8  	frame_type;		/* ֡���ͣ��ο� EN_FRAME_TYPE */
        sdk_uint32  frame_id;       /* ֡������ţ���Ƶ + ��Ƶ */
        sdk_uint32  frame_av_id;    /* ��Ƶ����Ƶ��֡����� */
        sdk_uint64  frame_ts;       /* ֡ʱ���, ��λms */
        sdk_uint32  frame_size;     /* ֡��С */
        sdk_uint32  frame_offset;   /* ֡ƫ��ֵ */
        sdk_uint8   *pdata;         /* ֡���� */
    }Dev_Stream_Frame_t;

    /* �澯��Ϣ���� */
    typedef struct Dev_Alarm_t
    {
        EN_ALARM_TYPE   type;           /* �澯���� */
        sdk_uint16      channel_index;  /* ͨ���� */
        sdk_uint8       status;         /* �澯״̬, �ο�"EN_ALARM_STATUS" */
    }Dev_Alarm_t;

    /* ƽ̨��Ϣ���� */
    typedef struct Entry_Serv_t
    {
        char        ip[64+1];   /* ��ڷ���IP */
        sdk_int16   port;       /* ��ڷ���Port */
    }Entry_Serv_t;

    /* �豸��Ϣ���� */
    typedef struct Dev_Info_t
    {
        char                dev_id[32+1];       /* �豸ID��ƽ̨�˵��豸Ψһ��ʶ����󳤶�32BYTE */
        sdk_uint8           dev_type;           /* �豸����: 1 ipc, 2 nvr, 3 dvr, 4 smartbox */
        Dev_OEM_Info_t      oem_info;
        Dev_Attribute_t     attr;
        sdk_int16           channel_num;        /* ͨ������ */
        Dev_Channel_Info_t* channel_list;       /* ͨ����Ϣ�б� */

        char                log_path[128+1];    /* ��־·�� */
        sdk_uint8           log_level;          /* ��־����: 0 fatal, 1 error, 2 warning, 3 info, 4 debug, 5 trace */
        sdk_uint32          log_max_size;       /* ��־���ֵ����λ KB, default:200 */
        sdk_uint8           log_backup_flag;    /* ��־���ݱ�־��0 ������ 1���ݣ� Ĭ��ֵ 0 */
    }Dev_Info_t;

    /**
    * ��ʼ��SDK
    * @param dev_info   �豸��ϸ��Ϣ
    */
	SDK_API int Dev_Sdk_Init(Entry_Serv_t* entry_serv, Dev_Info_t *dev_info);

    /**
    * ����SDK
    * @param
    */
	SDK_API void Dev_Sdk_Uninit(void);

    /**
    * ����ҵ������ָ��Ļص�����
    * @param cb_func    �ص�����
    * @param user_data  ҵ������ݣ�SDK�ص�ʱ����Ϊ�����ش���ҵ���
    */
	SDK_API void Dev_Sdk_Set_CB(Dev_Cmd_Cb_Func cb_func, void* user_data);

    /**
    * ����Ƶ֡�����ϱ��ӿ�
    * @param frame  ֡��Ϣ
    */
	SDK_API int Dev_Sdk_Stream_Frame_Report(Dev_Stream_Frame_t *frame);

    /**
    * �豸�澯�ϱ��ӿ�
    * @param alarm  �澯��Ϣ
    */
	SDK_API void Dev_Sdk_Alarm_Report(Dev_Alarm_t alarm);

    /**
    * ��ȡʱ����ӿ�
    * @return   ʱ�������λ: ms
    */
	SDK_API sdk_uint64 Dev_Sdk_Get_Timestamp(void);
     
    /**
    * ��ȡ�豸ID�ӿ�
    * @return   �豸ID�ڴ��ַ
    */
    SDK_API char* Dev_Sdk_Get_Device_ID(void);
	
    /**
    * ͨ��״̬�ϱ��ϱ��ӿ�
    * @param channel_index  ͨ��ID
    * @param status         ͨ��״̬
    * @return   ���ý��
    */
    SDK_API int Dev_Sdk_Channel_Status_Report(sdk_uint16 channel_index, Dev_Channel_Info_t* status);

    /**
    * ץ�ĵ�ͼƬ�ϱ��ӿ�
    * @param channel_index  ͨ��ID
    * @param pic_fmt        ͼƬ��ʽ
    * @param pic_data       ͼƬ���ݻ���
    * @param pic_size       ͼƬ���ݴ�С
    * @return   ���ý��
    */
    SDK_API int Dev_Sdk_Snap_Picture_Report(sdk_uint16 channel_index, EN_PIC_FMT_TYPE pic_fmt, sdk_uint8 *pic_data, sdk_uint32 pic_size );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __DEVICE_SDK_H__ */

