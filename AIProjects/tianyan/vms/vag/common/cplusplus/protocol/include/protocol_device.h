
#ifndef __PROTOCOL_DEVICE_H__
#define __PROTOCOL_DEVICE_H__

#include <vector>
#include "protocol_header.h"

using namespace std;

namespace protocol{
#pragma pack (push, 1)  
    
    //设备命令类型
    enum DeviceCmdType
    {
        CMD_PTZ             = 0x0001, // 云台控制命令
        CMD_REBOOT          = 0x0002, // 设备重启命令
        CMD_RESET           = 0x0003, // 设备重置命令
        CMD_DEV_MGR_UPDATE  = 0x0004, // 设备管理状态更新指令
    };
    
    //设备云台操作类型
    enum DevicePtzOpt
    {
        PTZ_OPT_STOP        = 0x0000,  //全部停止
        PTZ_OPT_PAN_LEFT    = 0x0001,  //水平向左
        PTZ_OPT_PAN_RIGHT   = 0x0002,  //水平向右
        PTZ_OPT_TILT_UP     = 0x0004,  //垂直向上
        PTZ_OPT_TILT_DOWN   = 0x0008,  //垂直向下
        PTZ_OPT_ZOOM_OUT    = 0x0010,  //变焦缩小
        PTZ_OPT_ZOOM_IN     = 0x0020,  //变焦放大
    };

    //云台控制命令数据定义
    struct PtzCmdData
    {
        uint16 opt_type;    //refer to 'DevicePtzOpt'
        uint8 param1;  // 参数1
        uint8 param2;  // 参数2
    };
    CDataStream& operator<<(CDataStream& ds, PtzCmdData& status);
    CDataStream& operator>>(CDataStream& ds, PtzCmdData& status);

    //设备管理状态变更通知数据定义
    enum EnDeviceMgrType
    {
        DEV_MGR_CH_ADD = 0x0001,    // 设备通道增加
        DEV_MGR_CH_MD  = 0x0002,    // 设备通道修改
        DEV_MGR_CH_DEL = 0x0003,    // 设备通道删除
    };
    struct DeviceMgrUpdateCmdData
    {
        uint16 mgr_type;    //refer to 'EnDeviceMgrType'
    };
    
    // 设备媒体会话状态定义
    struct DeviceMediaSessionStatus
    {
        string session_id;      // 会话ID
        uint16 session_type;    // refer to 'MediaSessionType'
		uint8 session_media;    // 媒体类型: 1:video, 2:audio, 3:all(video+audio)
        uint8 session_status;   // 媒体会话状态 0x00：会话结束 01：会话建立中 0x02：会话ok
        string device_id;       // 设备ID
        uint16 channel_id;      // 通道ID
        uint8  stream_id;       // 码流ID, refer to 'EnSteamType'
		HostAddr stream_addr;   // stream server address
    };
    CDataStream& operator<<(CDataStream& ds, DeviceMediaSessionStatus& status);
    CDataStream& operator>>(CDataStream& ds, DeviceMediaSessionStatus& status);
    
    //*************************************
    //**类型：设备登陆(MSG_ID_DEV_LOGIN)
    //**定义: DeviceLoginReq
    //**方向：Device->SMS
    //*************************************
    struct DeviceLoginReq
	{
		uint32 mask;

		//0x01
		string device_id;       // device id
        string version;
        uint8 dev_type;         // refer to 'EnDeviceType' 	
		uint16 channel_num;     // 总的通道数
		vector<DevChannelInfo> channels;   // 通道列表
        token_t token;
		
		//0x02
		HostAddr private_addr;      // 设备本地地址
	};
    CDataStream& operator<<(CDataStream& ds, DeviceLoginReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceLoginReq& req);
    
    struct DeviceLoginResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0X01
		HostAddr public_addr; // 设备公网地址
	};
    CDataStream& operator<<(CDataStream& ds, DeviceLoginResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceLoginResp& resp);
    
    //*************************************************
    //**类型：设备能力上报(MSG_ID_DEV_ABILITY_REPORT)
    //**定义: DeviceAbilityReportReq
    //**方向：Device->SMS
    //*************************************************
    struct DeviceAbilityReportReq
	{
        uint32 mask;
        
        //0x01 媒体能力
        uint8 media_trans_type;             // 媒体传输方式: 1-TCP, 2-UDP, 3-ALL(TCP、UDP)
        uint8 max_live_streams_per_ch;      // 一个通道支持最大实时码流数
        uint8 max_playback_streams_per_ch;  // 一个通道支持最大回放流数
        uint8 max_playback_streams;         // 一个设备支持的最大回放流数
        
        //0x02 
        uint32 disc_size;
        uint32 disc_free_size;
	};
    CDataStream& operator<<(CDataStream& ds, DeviceAbilityReportReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceAbilityReportReq& req);
    
    struct DeviceAbilityReportResp
    {
        uint32 mask;
		int32 resp_code;
    };
    CDataStream& operator<<(CDataStream& ds, DeviceAbilityReportResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceAbilityReportResp& resp);
    
    //*************************************************
    //**类型：设备状态上报(MSG_ID_DEV_STATUS_REPORT)
    //**定义: DeviceStatusReportReq
    //**方向：Device->SMS
    //*************************************************
    struct DeviceStatusReportReq
	{
        uint32 mask;
        
        //0x01
		uint8 channel_num;                  // 状态有变更的通道数目
        vector<DevChannelInfo> channels;    // 通道信息
		
		//0x02
		uint16 media_session_num;                           // 状态有变更的媒体会话数目
        vector<DeviceMediaSessionStatus> media_sessions;    // 会话信息
		
        //0x04
        uint8 sdcard_status;  // refer to 'Sdcard_Status'

	};
    CDataStream& operator<<(CDataStream& ds, DeviceStatusReportReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceStatusReportReq& req);
    
	struct DeviceStatusReportResp
	{
		uint32 mask;
        int32 resp_code;
		
		//0x01
		uint16 expected_cycle;
	};
    CDataStream& operator<<(CDataStream& ds, DeviceStatusReportResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceStatusReportResp& resp);
    
	//*************************************************
    //**类型：设备告警上报
    //**定义: DeviceAlarmReportReq
    //**方向：Device->SMS
    //*************************************************
    struct DeviceAlarmReportReq
	{
        uint32 mask;
        
        //0x01
		string device_id;
        uint16 channel_id;
        uint32 alarm_type;  // refer to 'EnAlarmType'
        uint8 alarm_status;	// refer to 'EnAlarmStatus'
        
        //0x02
        uint32 alarm_data_size;
        vector<uint8> alarm_datas;
	};
    CDataStream& operator<<(CDataStream& ds, DeviceAlarmReportReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceAlarmReportReq& req);
    
	struct DeviceAlarmReportResp
	{
		uint32 mask;
        int32 resp_code;
	};
    CDataStream& operator<<(CDataStream& ds, DeviceAlarmReportResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceAlarmReportResp& resp);
	
    //********************************************
    //**类型：设备图片上传(MSG_ID_DEV_PIC_UPLOAD)
    //**定义: DevicePictureUploadReq
    //**方向：SMS->Device
    //********************************************
    struct DevicePictureUploadReq
    {
        uint32 mask;
        
        //0x01
        string device_id;
        uint16 channel_id;
    };
    CDataStream& operator<<(CDataStream& ds, DevicePictureUploadReq& req);
    CDataStream& operator>>(CDataStream& ds, DevicePictureUploadReq& req);
    
    struct DevicePictureUploadResp
	{
		uint32 mask;
        int32 resp_code;
        
        //0x01
        string upload_url;
    };
    CDataStream& operator<<(CDataStream& ds, DevicePictureUploadResp& resp);
    CDataStream& operator>>(CDataStream& ds, DevicePictureUploadResp& resp);
    
    //********************************************
    //**类型：设备抓拍(MSG_ID_DEV_SNAP)
    //**定义: DeviceSnapReq
    //**方向：SMS->Device
    //********************************************
    struct DeviceSnapReq
    {
        uint32 mask;
        
        //0x01
        string device_id;
        uint16 channel_id;
        //string upload_url;
    };
    CDataStream& operator<<(CDataStream& ds, DeviceSnapReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceSnapReq& req);
    
    struct DeviceSnapResp
	{
		uint32 mask;
        int32 resp_code;

        //0x01
        string device_id;
        uint16 channel_id;
        string pic_fmt; //"bmp", "jpg", ...
        uint32 pic_size;

        //0x02
        uint32 offset;
        uint32 data_size;
        vector<uint8> datas;
    };
    CDataStream& operator<<(CDataStream& ds, DeviceSnapResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceSnapResp& resp);
    
	//********************************************
    //**类型：设备控制(MSG_ID_DEV_CTRL)
    //**定义: DeviceCtrlReq
    //**方向：SMS->Device
    //********************************************
    struct DeviceCtrlReq
    {
        uint32 mask;
        
        //0x01
        string device_id;
        uint16 channel_id;
        uint16 cmd_type;    //refer to 'DeviceCmdType'
		
		//0x02
        uint16 cmd_data_size;	 // 指令数据大小
        vector<uint8> cmd_datas; // 指令数据
    };
    CDataStream& operator<<(CDataStream& ds, DeviceCtrlReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceCtrlReq& req);
    
    struct DeviceCtrlResp
	{
		uint32 mask;
        int32 resp_code;
	};
    CDataStream& operator<<(CDataStream& ds, DeviceCtrlResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceCtrlResp& resp);
	
	//********************************************
    //**类型：设备参数设置(MSG_ID_DEV_PARAM_SET)
    //**定义: DeviceParamSetReq
    //**方向：SMS->Device
    //********************************************
    struct DeviceParamSetReq
    {
        uint32 mask;
        
        //0x01
        string device_id;
        uint16 channel_id;
        uint16 param_type;
		
		//0x02
        uint16 param_data_size;	// 参数数据大小
        vector<uint8> param_datas;   	// 参数数据
    };
    CDataStream& operator<<(CDataStream& ds, DeviceParamSetReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceParamSetReq& req);
	
    struct DeviceParamSetResp
	{
		uint32 mask;
        int32 resp_code;
	};
    CDataStream& operator<<(CDataStream& ds, DeviceParamSetResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceParamSetResp& resp);
	
	//********************************************
    //**类型：设备参数Get(MSG_ID_DEV_PARAM_GET)
    //**定义: DeviceParamGetReq
	//**方向：SMS->Device
    //********************************************
    struct DeviceParamGetReq
    {
        uint32 mask;
        
        //0x01
        string device_id;
        uint16 channel_id;
        uint16 param_type;
		
		//0x02
        uint16 param_data_size;	// 参数数据大小
        vector<uint8> param_datas;   	// 参数数据
    };
    CDataStream& operator<<(CDataStream& ds, DeviceParamGetReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceParamGetReq& req);
	
	struct DeviceParamGetResp
    {
        uint32 mask;
		int32 resp_code;
        
        //0x01
        string device_id;
        uint16 channel_id;
        uint16 param_type;
		
		//0x02
        uint16 param_data_size;	// 参数数据大小
        vector<uint8> param_datas;   	// 参数数据
    };
    CDataStream& operator<<(CDataStream& ds, DeviceParamGetResp& req);
    CDataStream& operator>>(CDataStream& ds, DeviceParamGetResp& req);
	
	//**********************************************************
    //**类型：查询设备录像列表(MSG_ID_DEV_RECORD_LIST_QUERY)
    //**定义: DeviceRecordListQueryReq
    //**方向：SMS->Device
    //**********************************************************
    struct DeviceRecordListQueryReq
    {
        uint32 mask;
        
        //0x01
        string device_id;
        uint16 channel_id;
        uint8  stream_id;   //码流ID, refer to 'EnSteamType'
        uint32 begin_time;
		uint32 end_time;
    };
    CDataStream& operator<<(CDataStream& ds, DeviceRecordListQueryReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceRecordListQueryReq& req);
    
    struct DeviceRecordListQueryResp
    {
        uint32 mask;
		int32 resp_code;
        
        //0x01
        string device_id;
        uint16 channel_id;
        uint8  stream_id;   //refer to 'EnSteamType'
        uint16 block_total_num;
        uint32 block_seq;
        uint16 block_num;
        vector<HistoryRecordBlock> record_blocks;
    };
    CDataStream& operator<<(CDataStream& ds, DeviceRecordListQueryResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceRecordListQueryResp& resp);
    
    
    //**********************************************
    //**类型：设备媒体打开(MSG_ID_DEV_MEDIA_OPEN)
    //**定义: DeviceMediaOpenReq
    //**方向：SMS-->Device
    //**********************************************
    struct DeviceMediaOpenReq
	{
        uint32 mask;
        
        //0x01 基本会话信息
		string device_id;
        uint16 channel_id;
        uint8  stream_id;       // 码流ID, refer to 'EnSteamType'
		uint16 session_type;    // refer to 'MediaSessionType'
        string session_id;      // 会话ID
        uint8 session_media;    // 媒体类型：0x01:Video 0x02:Audio, 0x03:all
        
        //0x02 视频格式
        VideoCodecInfo video_codec;
        
        //0x04 音频格式
        AudioCodecInfo audio_codec;
        
        //0x08 媒体传输方式
        uint8 transport_type;   // 媒体传输类型：1:ES_OVER_TCP 2:ES_OVER_UDP, 默认值是1
        
        //0x10 媒体时间
        uint32 begin_time;      // 开始时间(实时浏览时为0), UTC时间
        uint32 end_time;        // 结束时间(实时浏览时为0), UTC时间
        
        //0x20 流服务器列表信息
		uint16 stream_num;
        vector<HostAddr> stream_servers;
        token_t stream_token;
	};
    CDataStream& operator<<(CDataStream& ds, DeviceMediaOpenReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceMediaOpenReq& req);
    
	struct DeviceMediaOpenResp
    {
        uint32 mask;
		int32 resp_code;
        
        //0x01
        string device_id;
        uint16 channel_id;
        uint8  stream_id;       // refer to 'EnSteamType'
		string session_id;      // 会话ID
        HostAddr stream_server; //设备连接的流服务器
    };
    CDataStream& operator<<(CDataStream& ds, DeviceMediaOpenResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceMediaOpenResp& resp);
    
    //***********************************************
    //**类型：设备媒体关闭请求(MSG_ID_DEV_MEDIA_CLOSE)
    //**定义: DeviceMediaCloseReq
    //**方向：SMS->Device
    //***********************************************
    struct DeviceMediaCloseReq
	{
        uint32 mask;
        
        //0x01 基本会话信息
        string device_id;
        uint16 channel_id;
        uint8  stream_id;       // refer to 'EnSteamType'
		string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, DeviceMediaCloseReq& req);
    CDataStream& operator>>(CDataStream& ds, DeviceMediaCloseReq& req);
    
    struct DeviceMediaCloseResp
    {
        uint32 mask;
		int32 resp_code;
    };
    CDataStream& operator<<(CDataStream& ds, DeviceMediaCloseResp& resp);
    CDataStream& operator>>(CDataStream& ds, DeviceMediaCloseResp& resp);
#pragma pack (pop)	
};

#endif
