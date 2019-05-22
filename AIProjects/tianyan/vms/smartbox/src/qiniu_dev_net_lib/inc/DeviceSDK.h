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

    /* 设备类型定义 */
    typedef enum en_device_type
    {
        EN_DEV_TYPE_IPC     = 0,
        EN_DEV_TYPE_NVR     = 1,
        EN_DEV_TYPE_DVR     = 2,
        EN_DEV_TYPE_QNSMB   = 3     /* 七牛智能盒子（QiNiu SmartBox）*/
    }EN_DEV_TYPE;

    /* 码流类型定义 */
    typedef enum en_stream_type
    {
        EN_STREAM_TYPE_MAIN     = 0,    /* 主码流  */
        EN_STREAM_TYPE_SUB01    = 1     /* 子码流1 */
    }EN_STREAM_TYPE;

    /* 通道状态定义 */
    typedef enum en_channel_status
    {
        EN_CH_STS_OFFLINE   = 0,    /* 通道状态下线  */
        EN_CH_STS_ONLINE    = 1     /* 通道状态上线  */
    }EN_CH_STATUS;
     
    /* 帧类型定义 */
    typedef enum en_frame_type
    {
        EN_FRM_TYPE_I   = 0,    /* I帧    */
        EN_FRM_TYPE_P   = 1,    /* P帧    */
        EN_FRM_TYPE_AU  = 2     /* 音频帧 */
    }EN_FRAME_TYPE;
    
    /* 图片格式类型定义 */
    typedef enum en_pic_fmt_type
    {
        EN_PIC_FMT_BMP  = 0,
        EN_PIC_FMT_JPEG = 1
    }EN_PIC_FMT_TYPE;

    /* 视频编码格式类型定义 */
    typedef enum en_video_fmt_type
    {
        EN_VIDEO_FMT_H264   = 0,
        EN_VIDEO_FMT_H265   = 1
    }EN_VIDEO_FMT_TYPE;

    /* 音频编码格式类型定义 */
    typedef enum en_audio_fmt_type
    {
        EN_AUDIO_FMT_AAC    = 0,
        EN_AUDIO_FMT_G711_A = 1,
        EN_AUDIO_FMT_G711_U = 2,
        EN_AUDIO_FMT_MP3    = 3
    }EN_AUDIO_FMT_TYPE;

    /* 音频通道类型定义 */
    typedef enum en_audio_ch_type
    {
        EN_AUDIO_CH_MONO    = 0,    /* 单声道 */
        EN_AUDIO_CH_STEREO  = 1     /* 立体声道 */
    }EN_AUDIO_CH_TYPE;

    /* 音频位宽类型定义 */
    typedef enum en_audio_bitwidth_type
    {
        EN_AUDIO_BW_8BIT    = 0,    /* 位宽：8bit */
        EN_AUDIO_BW_16BIT   = 1     /* 位宽：16bit */
    }EN_AUDIO_BW_TYPE;

    /* 音频采样率类型定义 */
    typedef enum en_audio_sr_type
    {
        EN_AUDIO_SR_8_KHZ       = 0,    /* 采样率：8khz */
        EN_AUDIO_SR_11_025_KHZ  = 1,    /* 采样率：11.025khz */
        EN_AUDIO_SR_12_KHZ      = 2,    /* 采样率：12khz */
        EN_AUDIO_SR_16_KHZ      = 3,    /* 采样率：16khz */
        EN_AUDIO_SR_22_05_KHZ   = 4,    /* 采样率：22.05khz */
        EN_AUDIO_SR_24_KHZ      = 5,    /* 采样率：24khz */
        EN_AUDIO_SR_32_KHZ      = 6,    /* 采样率：32khz */
        EN_AUDIO_SR_44_1_KHZ    = 7,    /* 采样率：44.1khz */
        EN_AUDIO_SR_48_KHZ      = 8,    /* 采样率：48khz */
        EN_AUDIO_SR_64_KHZ      = 9,    /* 采样率：64khz */
        EN_AUDIO_SR_88_2_KHZ    = 10,   /* 采样率：88.2khz */
        EN_AUDIO_SR_96_KHZ      = 11    /* 采样率：96khz */
    }EN_AUDIO_SR_TYPE;

    /* 设备告警类型定义 */
    typedef enum en_alarm_type
    {
        EN_ALARM_MOVE_DETECT    = 0x00010001,   /* 移动侦测 */
        EN_ALARM_VIDEO_LOST     = 0x00010002,   /* 视频丢失 */
        EN_ALARM_VIDEO_SHELTER  = 0x00010003,   /* 视频遮挡 */	
        EN_ALARM_VIDEO_DISC     = 0x00010004,   /* 磁盘告警 */	
        EN_ALARM_VIDEO_RECORD   = 0x00010005    /* 前端录像告警 */
    }EN_ALARM_TYPE;

    /* 告警状态定义 */
    typedef enum en_alarm_status
    {
        EN_ALARM_STS_CLEAN      = 0,   /* 告警消去 */
        EN_ALARM_STS_ACTIVE     = 1,   /* 告警开启 */
        EN_ALARM_STS_KEEP       = 2    /* 告警保持 */
    }EN_ALARM_STATUS;

    /* 控制指令类型定义 */
    typedef enum en_cmd_type
    {
        EN_CMD_LIVE_OPEN,   /* 打开实时视频, cmd_agrs-->stream_id(uint8) [0:main_stream, 1:sub_stream] */
        EN_CMD_LIVE_CLOSE,  /* 关闭实时视频 */
        EN_CMD_SNAP,        /* 抓拍图片 */
        EN_CMD_PTZ,         /* 云台控制 */
        EN_CMD_PARAM_GET,   /* 获取设备参数 */
        EN_CMD_PARAM_SET,   /* 设置设备参数 */
        EN_CMD_MGR_UPDATE   /* 更新设备管理状态, cmd_agrs-->mgr_type(uint16) [ 1:add_channel, 2:modify_channel, 3: del_channel ] */
    }EN_CMD_TYPE;

    /* 控制指令参数定义 */
    typedef struct Dev_Cmd_Param_t
    {
        sdk_uint16  cmd_type;       /* 命令类型，参考EN_CMD_TYPE */
        int         channel_index;  /* 通道 */
        char        cmd_args[512];	/* 命令参数，具体参数参见文档 */
    }Dev_Cmd_Param_t;

    /* 业务层接受sdk指令的回调函数原型定义 */
    typedef void(*Dev_Cmd_Cb_Func)(void* user_data, Dev_Cmd_Param_t* cmd);

    /* 音频编解码信息定义 */
    typedef struct Dev_Audio_Codec_Info_t
    {
        sdk_uint8   codec_fmt;      /* 音频格式, 参考 "EN_AUDIO_FMT_TYPE" */
        sdk_uint8   channel;        /* 音频通道, 参考 "EN_AUDIO_CH_TYPE" */
        sdk_uint8   sample;         /* 音频采样率, 参考 "EN_AUDIO_SR_TYPE" */
        sdk_uint8   bitwidth;       /* 位宽, 参考 "EN_AUDIO_BW_TYPE" */
        sdk_uint8   sepc_size;      /* 音频详细信息长度 */
        sdk_uint8   sepc_data[8];   /* 音频详细信息,具体参考文档 */
    }Dev_Audio_Codec_Info_t;

    /* 视频编解码信息定义 */
    typedef struct Dev_Video_Codec_Info_t
    {
        sdk_uint8   codec_fmt;    /* 音频格式, 参考 "EN_VIDEO_FMT_TYPE" */
    }Dev_Video_Codec_Info_t;

    /* 设备属性定义 */
	typedef struct Dev_Attribute_t
	{
        sdk_uint8   has_microphone;   /* 是否有拾音器: 0 没有, 1 有 */
		sdk_uint8   has_hard_disk;    /* 是否有硬盘:   0 没有, 1 有 */
        sdk_uint8   can_recv_audio;   /* 可以接受音频: 0 不支持, 1 支持 */
	}Dev_Attribute_t;

    /* 设备OEM信息定义 */
    typedef struct Dev_OEM_Info_t
    {
        sdk_int32   OEMID;              /* OEM ID */
        char        OEM_name[2+1];      /* 厂商OEM名称,具体根据不同厂商由平台统一提供 */
        char        MAC[17 + 1];        /* MAC 地址 */
        char        SN[16+1];           /* 序列号 */			    
        char        Model[64 + 1];      /* 设备型号 */
        char        Factory[128 + 1];   /* 厂商名称(eg. 海康威视、大华等)*/
    }Dev_OEM_Info_t;

    /* 通道的码流信息定义 */
    typedef struct Dev_Stream_Info_t
    {
        sdk_uint8               stream_id;      /* 码流ID: 0 主码流, 1 子码流 */
        sdk_uint32              video_height;   /* 视频高度 */
        sdk_uint32              video_width;    /* 视频宽度 */
        Dev_Video_Codec_Info_t  video_codec;    /* 视频编码信息 */
    }Dev_Stream_Info_t;

    /* 设备的通道信息定义 */
    typedef struct Dev_Channel_Info_t
    {
        sdk_uint16              channel_index;  /* 通道ID，起始ID从1开始 */
        sdk_uint16              channel_type;   /* 通道类型, 0: 摄像头 */
        sdk_uint8               channel_status; /* 通道状态, 参考"EN_CH_STATUS" */
        sdk_uint8               has_ptz;        /* 是否有云台: 0 没有, 1 有 */
        sdk_uint8               stream_num;     /* 码流数量 */
        Dev_Stream_Info_t*      stream_list;    /* 码流信息列表 */
        Dev_Audio_Codec_Info_t  adudo_codec;    /* 音频编码信息 */
    }Dev_Channel_Info_t;

    /* 帧信息定义 */
    typedef struct Dev_Stream_Frame_t
    {
        sdk_uint16  channel_index;  /* 通道号 */
        sdk_uint8  	stream_id;      /* 码率ID: 0 主码流, 1 子码流 */
        sdk_uint8  	frame_type;		/* 帧类型，参考 EN_FRAME_TYPE */
        sdk_uint32  frame_id;       /* 帧号总序号：视频 + 音频 */
        sdk_uint32  frame_av_id;    /* 音频、视频的帧号序号 */
        sdk_uint64  frame_ts;       /* 帧时间戳, 单位ms */
        sdk_uint32  frame_size;     /* 帧大小 */
        sdk_uint32  frame_offset;   /* 帧偏移值 */
        sdk_uint8   *pdata;         /* 帧数据 */
    }Dev_Stream_Frame_t;

    /* 告警信息定义 */
    typedef struct Dev_Alarm_t
    {
        EN_ALARM_TYPE   type;           /* 告警类型 */
        sdk_uint16      channel_index;  /* 通道号 */
        sdk_uint8       status;         /* 告警状态, 参考"EN_ALARM_STATUS" */
    }Dev_Alarm_t;

    /* 平台信息定义 */
    typedef struct Entry_Serv_t
    {
        char        ip[64+1];   /* 入口服务IP */
        sdk_int16   port;       /* 入口服务Port */
    }Entry_Serv_t;

    /* 设备信息定义 */
    typedef struct Dev_Info_t
    {
        char                dev_id[32+1];       /* 设备ID，平台端的设备唯一标识，最大长度32BYTE */
        sdk_uint8           dev_type;           /* 设备类型: 1 ipc, 2 nvr, 3 dvr, 4 smartbox */
        Dev_OEM_Info_t      oem_info;
        Dev_Attribute_t     attr;
        sdk_int16           channel_num;        /* 通道数量 */
        Dev_Channel_Info_t* channel_list;       /* 通道信息列表 */

        char                log_path[128+1];    /* 日志路径 */
        sdk_uint8           log_level;          /* 日志级别: 0 fatal, 1 error, 2 warning, 3 info, 4 debug, 5 trace */
        sdk_uint32          log_max_size;       /* 日志最大值：单位 KB, default:200 */
        sdk_uint8           log_backup_flag;    /* 日志备份标志：0 不备份 1备份， 默认值 0 */
    }Dev_Info_t;

    /**
    * 初始化SDK
    * @param dev_info   设备详细信息
    */
	SDK_API int Dev_Sdk_Init(Entry_Serv_t* entry_serv, Dev_Info_t *dev_info);

    /**
    * 结束SDK
    * @param
    */
	SDK_API void Dev_Sdk_Uninit(void);

    /**
    * 设置业务层接受指令的回调函数
    * @param cb_func    回调函数
    * @param user_data  业务层数据，SDK回调时会作为参数回传给业务层
    */
	SDK_API void Dev_Sdk_Set_CB(Dev_Cmd_Cb_Func cb_func, void* user_data);

    /**
    * 音视频帧数据上报接口
    * @param frame  帧信息
    */
	SDK_API int Dev_Sdk_Stream_Frame_Report(Dev_Stream_Frame_t *frame);

    /**
    * 设备告警上报接口
    * @param alarm  告警信息
    */
	SDK_API void Dev_Sdk_Alarm_Report(Dev_Alarm_t alarm);

    /**
    * 获取时间戳接口
    * @return   时间戳，单位: ms
    */
	SDK_API sdk_uint64 Dev_Sdk_Get_Timestamp(void);
     
    /**
    * 获取设备ID接口
    * @return   设备ID内存地址
    */
    SDK_API char* Dev_Sdk_Get_Device_ID(void);
	
    /**
    * 通道状态上报上报接口
    * @param channel_index  通道ID
    * @param status         通道状态
    * @return   调用结果
    */
    SDK_API int Dev_Sdk_Channel_Status_Report(sdk_uint16 channel_index, Dev_Channel_Info_t* status);

    /**
    * 抓拍的图片上报接口
    * @param channel_index  通道ID
    * @param pic_fmt        图片格式
    * @param pic_data       图片数据缓存
    * @param pic_size       图片数据大小
    * @return   调用结果
    */
    SDK_API int Dev_Sdk_Snap_Picture_Report(sdk_uint16 channel_index, EN_PIC_FMT_TYPE pic_fmt, sdk_uint8 *pic_data, sdk_uint32 pic_size );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __DEVICE_SDK_H__ */

