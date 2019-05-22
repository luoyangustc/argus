
#include "protocol_status.h"
namespace protocol{

    CDataStream& operator<<(CDataStream& ds, SDeviceSessionStatus& status)
    {
        ds << status.mask;
        
        if( status.mask & 0x01 )
        {
            ds.writestring(status.did.c_str());
        }

        if( status.mask & 0x02 )
        {
            ds << status.status;
            ds << status.timestamp;
        }

        if( status.mask & 0x04 )
        {
            ds.writestring(status.version.c_str());
            ds << status.dev_type;
            ds << status.channel_num;
            ds << status.session_serv_addr; 
        }

        if( status.mask & 0x08 )
        {
            ds << status.channel_list_size;
            for (int i = 0; i< status.channel_list_size; ++i)
            {
                ds << status.channel_list[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, SDeviceSessionStatus& status)
    {
        ds >> status.mask;
        
        if( status.mask & 0x01 )
        {
            status.did = ds.readstring();
        }

        if( status.mask & 0x02 )
        {
            ds >> status.status;
            ds >> status.timestamp;
        }

        if( status.mask & 0x04 )
        {
            status.version = ds.readstring();
            ds >> status.dev_type;
            ds >> status.channel_num;
            ds >> status.session_serv_addr; 
        }

        if( status.mask & 0x08 )
        {
            ds >> status.channel_list_size;
            for (int i = 0; i< status.channel_list_size; ++i)
            {
                DevChannelInfo channel;
                ds >> channel;
                status.channel_list.push_back(channel);
            }
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, SDeviceStreamStatus& status)
    {
        ds << status.mask;
        
        if( status.mask & 0x01 )
        {
            ds.writestring(status.session_id.c_str());
            ds << status.session_type;
            ds.writestring(status.did.c_str());
            ds << status.channel_id;
            ds << status.stream_id;
            ds << status.status;
            ds << status.timestamp;
            ds << status.stream_serv_addr;
        }

        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, SDeviceStreamStatus& status)
    {
        ds >> status.mask;
        
        if( status.mask & 0x01 )
        {
            status.session_id = ds.readstring();
            ds >> status.session_type;
            status.did = ds.readstring();
            ds >> status.channel_id;
            ds >> status.stream_id;
            ds >> status.status;
            ds >> status.timestamp;
            ds >> status.stream_serv_addr;
        }

        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, StsLoginReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds << req.ep_type;
            ds << req.http_port; 
            ds << req.serv_port;
        }
        
        if(req.mask&0x02)
        {
            ds << req.listen_ip_num;
            for(uint32 i=0; i<req.listen_ip_num;++i)
            {
                ds.writestring(req.listen_ips[i].c_str());
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StsLoginReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            ds >> req.ep_type;
            ds >> req.http_port; 
            ds >> req.serv_port;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.listen_ip_num;
            for(uint32 i=0; i<req.listen_ip_num;++i)
            {
                string strIp = ds.readstring();
                req.listen_ips.push_back(strIp);
            }
        }

        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StsLoginResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds << resp.load_expected_cycle;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StsLoginResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds >> resp.load_expected_cycle;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StsLoadReportReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds << req.tcp_conn_num;
            ds << req.cpu_use; 
            ds << req.memory_use;
        }
        
        if(req.mask&0x02)
        {
            ds << req.ip_num;
            std::map<string, SBandwidthInfo>::iterator it = req.ip_bw_infos.begin(); 
            for(; it!=req.ip_bw_infos.end();++it)
            {
                ds.writestring(it->first.c_str());
                ds << it->second;
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StsLoadReportReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            ds >> req.tcp_conn_num;
            ds >> req.cpu_use; 
            ds >> req.memory_use;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.ip_num;
            for(uint32 i=0; i<req.ip_num;++i)
            {
                string strIp;
                SBandwidthInfo bw;
                strIp = ds.readstring();
                ds >> bw;
                req.ip_bw_infos[strIp] = bw;
            }
        }

        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StsLoadReportResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds << resp.load_expected_cycle;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StsLoadReportResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds >> resp.load_expected_cycle;
        }
        
        return ds;
    }

	CDataStream& operator<<(CDataStream& ds, StsSessionStatusReportReq& req)
	{
		ds << req.mask;	
		if (req.mask & 0x01)
		{
			ds << req.device_num;
            for (std::vector<SDeviceSessionStatus>::iterator iter=req.devices.begin(); 
				iter != req.devices.end(); ++iter)
			{
				ds << *iter;
			}
		}
		return ds;
	}
	
	CDataStream& operator>>(CDataStream& ds, StsSessionStatusReportReq& req)
	{
		ds >> req.mask;	
		if (req.mask & 0x01)
		{
			ds >> req.device_num;
            for (int i=0; i<req.device_num; i++)
			{
                SDeviceSessionStatus status;
				ds >> status;
				req.devices.push_back(status);
			}
		}
		return ds;
	}
	
	CDataStream& operator<<(CDataStream& ds, StsSessionStatusReportResp& resp)
	{
		ds << resp.mask;		
		ds << resp.resp_code;
		return ds;
	}
	
	CDataStream& operator>>(CDataStream& ds, StsSessionStatusReportResp& resp)
	{
		ds >> resp.mask;		
		ds >> resp.resp_code;
        return ds;		
	}	
	
	CDataStream& operator<<(CDataStream& ds, StsStreamStatusReportReq& req)
	{
		ds << req.mask;	
		if (req.mask & 0x01)
		{
			ds << req.device_num;
			SDeviceStreamStatus status;
            for (std::vector<SDeviceStreamStatus>::iterator iter=req.devices.begin(); 
				iter != req.devices.end(); ++iter)
			{
				ds << *iter;
			}
		}
		return ds;
	}
	
	CDataStream& operator>>(CDataStream& ds, StsStreamStatusReportReq& req)
	{
		ds >> req.mask;	
		if (req.mask & 0x01)
		{
			ds >> req.device_num;
            for (int i=0; i<req.device_num; i++)
			{
                SDeviceStreamStatus status;
				ds >> status;
				req.devices.push_back(status);
			}
		}
		return ds;
	}
	
	CDataStream& operator<<(CDataStream& ds, StsStreamStatusReportResp& resp)
	{
		ds << resp.mask;		
		ds << resp.resp_code;
		return ds;
	}
	
	CDataStream& operator>>(CDataStream& ds, StsStreamStatusReportResp& resp)
	{
		ds >> resp.mask;		
		ds >> resp.resp_code;
		return ds;
	}
}



