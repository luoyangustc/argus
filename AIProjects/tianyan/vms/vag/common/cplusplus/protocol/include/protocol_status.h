
#ifndef __PROTOCOL_STATUS_H__
#define __PROTOCOL_STATUS_H__

#include <map>
#include <vector>
#include "protocol_header.h"
using namespace std;

namespace protocol{

#pragma pack (push, 1)

    //设备会话状态
    struct SDeviceSessionStatus
    {
        uint32 mask;
        enum
        {
            enm_dev_status_offline  = 0,
            enm_dev_status_online   = 1,
        };

        //0x01
        string did;
        
        //0x02
        uint8 status;
        STimeVal timestamp;

        //0x04
        string version;
        uint8 dev_type;
        uint16 channel_num;
        HostAddr session_serv_addr;

        //0x08
        uint16 channel_list_size;               // 通道数
		vector<DevChannelInfo> channel_list;    // 通道状态列表, 可变长
    };
    CDataStream& operator<<(CDataStream& ds, SDeviceSessionStatus& status);
    CDataStream& operator>>(CDataStream& ds, SDeviceSessionStatus& status);
    
    //设备流状态
    struct SDeviceStreamStatus
    {
        uint32 mask;
        enum
        {
            enm_dev_media_connected    = 1, //设备连接上
            enm_dev_media_disconnect   = 2, //设备主动断连
            enm_dev_media_closed       = 3  //设备被动断开连接
        };

        //0x01
        string session_id;
        uint16 session_type;
        string did;
        uint16 channel_id;
        uint8 stream_id;
        uint8 status;
        STimeVal timestamp;
        HostAddr stream_serv_addr;
    };
    CDataStream& operator<<(CDataStream& ds, SDeviceStreamStatus& status);
    CDataStream& operator>>(CDataStream& ds, SDeviceStreamStatus& status);

    /*************************************
    **类型：登陆接口(MSG_ID_STS_LOGIN)
    **定义: StsLoginReq/Resp
    **方向：SMS/STREAM->STATUS
    **************************************/
    struct StsLoginReq
	{
		uint32 mask;
        
        //0x01 
        uint16 ep_type;     //refer to 'EndPointType'
        uint16 http_port;
        uint16 serv_port;
        
        //0x02
		uint8 listen_ip_num;
        vector<string> listen_ips;
	};
    CDataStream& operator<<(CDataStream& ds, StsLoginReq& req);
    CDataStream& operator>>(CDataStream& ds, StsLoginReq& req);
    
    struct StsLoginResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		uint16 load_expected_cycle;
	};
    CDataStream& operator<<(CDataStream& ds, StsLoginResp& resp);
    CDataStream& operator>>(CDataStream& ds, StsLoginResp& resp);
    
    /*************************************
    **类型：负载上报接口(MSG_ID_STS_LOAD_REPORT)
    **定义: StsLoadReportReq/Resp
    **方向：SMS/STREAM->STATUS
    **************************************/
    struct StsLoadReportReq
	{
		uint32 mask;
        
        //0x01
        uint16 tcp_conn_num;    //tcp链接数
        uint16 cpu_use;         //cpu使用率
		uint16 memory_use;      //内存使用率
        
        //0x02
        uint8 ip_num;
        map<string, SBandwidthInfo> ip_bw_infos;
	};
    CDataStream& operator<<(CDataStream& ds, StsLoadReportReq& req);
    CDataStream& operator>>(CDataStream& ds, StsLoadReportReq& req);
    
    struct StsLoadReportResp
	{
		uint32 mask;
		int32 resp_code;
		
		//0x01
		uint16 load_expected_cycle;
	};
    CDataStream& operator<<(CDataStream& ds, StsLoadReportResp& resp);
    CDataStream& operator>>(CDataStream& ds, StsLoadReportResp& resp);
    
    /*************************************
    **类型：会话状态上报接口(MSG_ID_STS_SESSION_STATUS_REPORT)
    **定义: StsSessionStatusReportReq/Resp
    **方向：SMS->STATUS
    **************************************/
    struct StsSessionStatusReportReq
	{
		uint32 mask;
        
        //0x01
		uint32 device_num;
		std::vector<SDeviceSessionStatus> devices;  //会话服务：参考'SDeviceSessionStatus'
	};
    CDataStream& operator<<(CDataStream& ds, StsSessionStatusReportReq& req);
    CDataStream& operator>>(CDataStream& ds, StsSessionStatusReportReq& req);
    
    struct StsSessionStatusReportResp
	{
		uint32 mask;
		int32 resp_code;
	};
    CDataStream& operator<<(CDataStream& ds, StsSessionStatusReportResp& resp);
    CDataStream& operator>>(CDataStream& ds, StsSessionStatusReportResp& resp);
	
    /*************************************
    **类型：流状态上报接口(MSG_ID_STS_STREAM_STATUS_REPORT)
    **定义: StsStreamStatusReportReq/Resp
    **方向：STREAM->STATUS
    **************************************/
    struct StsStreamStatusReportReq
	{
		uint32 mask;
        
        //0x01
		uint32 device_num;
		std::vector<SDeviceStreamStatus> devices;  //流服务：参考'SDeviceStreamStatus'
	};
    CDataStream& operator<<(CDataStream& ds, StsStreamStatusReportReq& req);
    CDataStream& operator>>(CDataStream& ds, StsStreamStatusReportReq& req);
    
    struct StsStreamStatusReportResp
	{
		uint32 mask;
		int32 resp_code;
	};
    CDataStream& operator<<(CDataStream& ds, StsStreamStatusReportResp& resp);
    CDataStream& operator>>(CDataStream& ds, StsStreamStatusReportResp& resp);	
#pragma pack (pop)    
};

#endif
