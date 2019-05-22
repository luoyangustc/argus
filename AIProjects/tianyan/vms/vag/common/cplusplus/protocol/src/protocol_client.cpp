
#include "protocol_client.h"
namespace protocol{

    CDataStream& operator<<(CDataStream& ds, CuMediaSessionStatus& status)
    {
        ds.writestring(status.session_id.c_str());
        ds << status.session_type;
        ds << status.session_media;
        ds << status.session_status;
        ds.writestring(status.device_id.c_str());
        ds << status.channel_id;
        ds << status.stream_id;
        ds << status.stream_route_table_size;
        for(uint16 i=0; i<status.stream_route_table_size; ++i)
        {
            ds << status.stream_route_tables[i];
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuMediaSessionStatus& status)
    {
        status.session_id = ds.readstring();
        ds >> status.session_type;
        ds >> status.session_media;
        ds >> status.session_status;
        status.device_id = ds.readstring();
        ds >> status.channel_id;
        ds >> status.stream_id;
        ds >> status.stream_route_table_size;
        for(uint16 i=0; i<status.stream_route_table_size; ++i)
        {
            HostAddr host;
            ds >> host;
            status.stream_route_tables.push_back(host);
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuLoginReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.user_name.c_str());
            ds << req.token;
        }
        
        if(req.mask&0x02)
        {
            ds.writestring(req.private_ip.c_str());
            ds << req.private_port;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuLoginReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            req.user_name = ds.readstring();
            ds >> req.token;
        }
        
        if(req.mask&0x02)
        {
            req.private_ip = ds.readstring();
            ds >> req.private_port;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuLoginResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds.writestring(resp.public_ip.c_str());
            ds << resp.public_port;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuLoginResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            resp.public_ip = ds.readstring();
            ds >> resp.public_port;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuStatusReportReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds << req.media_session_num;
            for(uint32 i=0; i<req.media_session_num;++i)
            {
                ds << req.media_sessions[i];
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuStatusReportReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            ds >> req.media_session_num;
            for(uint32 i=0; i<req.media_session_num;++i)
            {
                CuMediaSessionStatus session;
                ds >> session;
                req.media_sessions.push_back(session);
            }
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuStatusReportResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds << resp.expected_cycle;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuStatusReportResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds >> resp.expected_cycle;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuMediaOpenReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.stream_id;
            ds << req.session_type;
            ds << req.session_media;
        }
        
        if(req.mask&0x02)
        {
            ds << req.transport_type;
        }
        
        if(req.mask&0x04)
        {
            ds << req.begin_time;
            ds << req.end_time;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuMediaOpenReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.stream_id;
            ds >> req.session_type;
            ds >> req.session_media;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.transport_type;
        }
        
        if(req.mask&0x04)
        {
            ds >> req.begin_time;
            ds >> req.end_time;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuMediaOpenResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
            ds.writestring(resp.device_id.c_str());
            ds << resp.channel_id;
            ds << resp.stream_id;
        }
        
        if(resp.mask&0x02)
        {
            ds << resp.video_codec;
        }
        
        if(resp.mask&0x04)
        {
            ds << resp.audio_codec;
        }
        
        if(resp.mask&0x08)
        {
            ds << resp.stream_route_table_size;
            for(uint32 i=0;i<resp.stream_route_table_size;++i)
            {
                ds << resp.stream_route_tables[i];
            }
            ds << resp.stream_token;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuMediaOpenResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
            resp.device_id = ds.readstring();
            ds >> resp.channel_id;
            ds >> resp.stream_id;
        }
        
        if(resp.mask&0x02)
        {
            ds >> resp.video_codec;
        }
        
        if(resp.mask&0x04)
        {
            ds >> resp.audio_codec;
        }
        
        if(resp.mask&0x08)
        {
            ds >> resp.stream_route_table_size;
            for(uint32 i=0;i<resp.stream_route_table_size;++i)
            {
                HostAddr host;
                ds >> host;
                resp.stream_route_tables.push_back(host);
            }
            ds >> resp.stream_token;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuMediaCloseReq& req)
    {
        ds << req.mask;        
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.stream_id;
            ds.writestring(req.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuMediaCloseReq& req)
    {
        ds >> req.mask;        
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.stream_id;
            req.session_id = ds.readstring();
        }        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, CuMediaCloseResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CuMediaCloseResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        return ds;
    }
}



