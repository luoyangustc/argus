#ifndef __PROTOCOL_CLIENT_H__
#define __PROTOCOL_CLIENT_H__

#include "protocol_header.h"

namespace protocol{
    
#pragma pack (push, 1) 

    // 客户端媒体会话状态定义
    struct CuMediaSessionStatus
    {
        string session_id;      // 会话ID
        uint16 session_type;    // refer to 'MediaSessionType'
		uint8 session_media;    // 媒体类型: 1:video, 2:audio, 3:all(video+audio)
        uint8 session_status;   // 媒体会话状态 0x00：会话结束 01：会话建立中 0x02：会话ok
        string device_id;       // 设备ID
        uint16 channel_id;      // 设备通道号
        uint8  stream_id;       // refer to 'EnSteamType'
		uint16 stream_route_table_size; // 路由表大小
        vector<HostAddr> stream_route_tables; // 路由表信息 
    };
    CDataStream& operator<<(CDataStream& ds, CuMediaSessionStatus& status);
    CDataStream& operator>>(CDataStream& ds, CuMediaSessionStatus& status);
    
    /*************************************
    **类型：客户端登陆接口(MSG_ID_CU_LOGIN)
    **定义: CuLoginReq/Resp
    **方向：CU->SMS
    **************************************/
    struct CuLoginReq
	{
		uint32 mask;

		//0x01
		string user_name;   // 登陆用户名
		token_t token;      // 验证token
        
		//0x02
		string private_ip;      // 设备本地IP
        uint16 private_port;    // 设备本地Port
	};
    CDataStream& operator<<(CDataStream& ds, CuLoginReq& req);
    CDataStream& operator>>(CDataStream& ds, CuLoginReq& req);

    struct CuLoginResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0X01
		string public_ip;   // 客户端公网IP
		uint16 public_port;  // 客户端公网PORT
	};
    CDataStream& operator<<(CDataStream& ds, CuLoginResp& resp);
    CDataStream& operator>>(CDataStream& ds, CuLoginResp& resp);
    
    
    /*************************************************
    **类型：客户端状态上报接口(MSG_ID_CU_STATUS_REPORT)
    **定义: CuStatusReportReq/Resp
    **方向：CU->SMS
    *************************************************/
    struct CuStatusReportReq
	{
		uint32 mask;
        
        //0x01 流会话状态
        uint16 media_session_num;
        vector<CuMediaSessionStatus> media_sessions;
	};
    CDataStream& operator<<(CDataStream& ds, CuStatusReportReq& req);
    CDataStream& operator>>(CDataStream& ds, CuStatusReportReq& req);
    
	struct CuStatusReportResp
	{
		uint32 mask;
        int32 resp_code;
		
		//0x01
		uint16 expected_cycle;
	};
    CDataStream& operator<<(CDataStream& ds, CuStatusReportResp& resp);
    CDataStream& operator>>(CDataStream& ds, CuStatusReportResp& resp);
    
    /***************************************************
    **类型：客户端打开设备媒体接口(MSG_ID_CU_MEDIA_OPEN)
    **定义: CuMediaOpenReq/Resp
    **方向：CU->SMS
    ***************************************************/
    struct CuMediaOpenReq
	{
        uint32 mask;
        
        //0x01 基本会话信息
        string device_id;       // 设备ID
        uint16 channel_id;      // 通道号
        uint8  stream_id;       // refer to 'EnSteamType'
        uint16 session_type;    // refer to 'MediaSessionType'
		uint8 session_media;	//媒体类型：0x01:Video 0x02:Audio, 0x03:all
        
        //0x02 媒体体传输方式
        uint8 transport_type;   // 媒体传输类型：1:ES_OVER_TCP 2:ES_OVER_UDP, 默认值是1
        
        //0x04 媒体时间
        uint32 begin_time;      // 开始时间(实时浏览时为0), UTC时间
        uint32 end_time;        // 结束时间(实时浏览时为0), UTC时间
	};
    CDataStream& operator<<(CDataStream& ds, CuMediaOpenReq& req);
    CDataStream& operator>>(CDataStream& ds, CuMediaOpenReq& req);

    struct CuMediaOpenResp
	{
        uint32 mask;
        int32 resp_code;
        
        //0x01 基本会话信息
        string session_id;    	// 会话ID
        string device_id;       // 设备ID
        uint16 channel_id;      // 通道号
        uint8  stream_id;       // refer to 'EnSteamType'
        
        //0x02 视频类型
        VideoCodecInfo video_codec;
        
        //0x04 音频类型
        AudioCodecInfo audio_codec;

        //0x08 媒体路由信息
		uint16 stream_route_table_size;         // 流路由表大小
        vector<HostAddr> stream_route_tables;   // 流路由信息
        token_t stream_token;                   // 流服务验证token
	};
    CDataStream& operator<<(CDataStream& ds, CuMediaOpenResp& resp);
    CDataStream& operator>>(CDataStream& ds, CuMediaOpenResp& resp);
    
    
    /***********************************************
    **类型：客户端媒体关闭接口(MSG_ID_CU_MEDIA_CLOSE)
    **定义: CuMediaCloseReq/Resp
    **方向：CU->SMS
    ***********************************************/
    struct CuMediaCloseReq
	{
        uint32 mask;
        
        //0x01 基本会话信息
        string device_id;
        uint16 channel_id;
        uint8  stream_id;       // refer to 'EnSteamType'
        string session_id;      // 会话ID
	};
    CDataStream& operator<<(CDataStream& ds, CuMediaCloseReq& req);
    CDataStream& operator>>(CDataStream& ds, CuMediaCloseReq& req);
    
    struct CuMediaCloseResp
	{
        uint32 mask;
        int32 resp_code;
	};
    CDataStream& operator<<(CDataStream& ds, CuMediaCloseResp& resp);
    CDataStream& operator>>(CDataStream& ds, CuMediaCloseResp& resp);
    
    
    /********************************************
    **类型：设备抓拍(MSG_ID_DEV_SNAP)
    **定义: DeviceSnapReq/Resp(refer to protocol_device.h)
    **方向：SMS->Device
    ********************************************/
    
    
	/********************************************
    **类型：设备控制(MSG_ID_DEV_CTRL)
    **定义: DeviceCtrlReq/Resp(refer to protocol_device.h)
    **方向：SMS->Device
    ********************************************/
    
	
	/********************************************
    **类型：设备参数设置(MSG_ID_DEV_PARAM_SET)
    **定义: DeviceParamSetReq/Resp(refer to protocol_device.h)
    **方向：SMS->Device
    ********************************************/
    
	
	/********************************************
    **类型：设备参数Get(MSG_ID_DEV_PARAM_GET)
    **定义: DeviceParamGetReq/Resp(refer to protocol_device.h)
	**方向：SMS->Device
    ********************************************/
    
	
	/**********************************************************
    **类型：查询设备录像列表(MSG_ID_DEV_RECORD_LIST_QUERY)
    **定义: DeviceHistoryListQueryReq/Resp(refer to protocol_device.h)
    **方向：SMS->Device
    **********************************************************/

#pragma pack (pop)    
    
};

#endif


