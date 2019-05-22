
#include "protocol_stream.h"
namespace protocol{

    CDataStream& operator<<(CDataStream& ds, StreamMediaConnectReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds << req.session_type;
            ds.writestring(req.session_id.c_str());
            ds << req.session_media;
            ds.writestring(req.endpoint_name.c_str());
            ds << req.endpoint_type;
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.stream_id; 
            ds << req.token;
        }
        
        if(req.mask&0x02)
        {
            ds << req.video_direct;
            ds << req.video_codec;
        }
        
        if(req.mask&0x04)
        {
            ds << req.audio_direct;
            ds << req.audio_codec;
        }
        
        if(req.mask&0x08)
        {
            ds << req.begin_time;
            ds << req.end_time;
        }
        
        if(req.mask&0x10)
        {
            ds << req.route_table_size;
            for(uint32 i=0; i<req.route_table_size;++i)
            {
                ds << req.route_tables[i];
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaConnectReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            ds >> req.session_type;
            req.session_id = ds.readstring();
            ds >> req.session_media;
            req.endpoint_name = ds.readstring();
            ds >> req.endpoint_type;
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.stream_id;
            ds >> req.token;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.video_direct;
            ds >> req.video_codec;
        }

        if(req.mask&0x04)
        {
            ds >> req.audio_direct;
            ds >> req.audio_codec;
        }

        if(req.mask&0x08)
        {
            ds >> req.begin_time;
            ds >> req.end_time;
        }
        
        if(req.mask&0x10)
        {
            ds >> req.route_table_size;
            for(uint32 i=0; i<req.route_table_size;++i)
            {
                HostAddr host;
                ds >> host;
                req.route_tables.push_back(host);
            }
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaConnectResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaConnectResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaDisconnectReq& req)
    {
        ds << req.mask;        
        if(req.mask&0x01)
        {
            ds.writestring(req.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaDisconnectReq& req)
    {
        ds >> req.mask;        
        if(req.mask&0x01)
        {
            req.session_id = ds.readstring();
        }        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaDisconnectResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaDisconnectResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaPlayReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.session_id.c_str());
        }
        
        if(req.mask&0x02)
        {
            ds << req.begin_time;
            ds << req.end_time;
        }
            
        if(req.mask&0x04)
        {
            ds << req.speed;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaPlayReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            req.session_id = ds.readstring();
        }
        
        if(req.mask&0x02)
        {
            ds >> req.begin_time;
            ds >> req.end_time;
        }
            
        if(req.mask&0x04)
        {
            ds >> req.speed;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaPlayResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaPlayResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaPauseReq& req)
    {
        ds << req.mask;        
        if(req.mask&0x01)
        {
            ds.writestring(req.session_id.c_str());
        }     
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaPauseReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            req.session_id = ds.readstring();
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaPauseResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaPauseResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
        }
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, StreamMediaCmdReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.session_id.c_str());
            ds << req.cmd_type;
        }

        if(req.mask&0x02)
        {
            ds << req.param_data_size;
            for(uint32 i=0; i<req.param_data_size; ++i)
            {
                ds << req.param_datas[i];
            }
        }

        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaCmdReq& req)
    {
        ds >> req.mask;

        if(req.mask&0x01)
        {
            req.session_id = ds.readstring();
            ds >> req.cmd_type;
        }

        if(req.mask&0x02)
        {
            ds >> req.param_data_size;
            for(uint32 i=0; i<req.param_data_size; ++i)
            {
                uint8 data;
                ds >> data;
                req.param_datas.push_back(data);
            }
        }

        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, StreamMediaCmdResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaCmdResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
        }
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, StreamMediaStatusReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.session_id.c_str());
        }
        
        if(req.mask&0x02)
        {
            ds << req.video_status;
        }
            
        if(req.mask&0x04)
        {
            ds << req.audio_status;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaStatusReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            req.session_id = ds.readstring();
        }
        
        if(req.mask&0x02)
        {
            ds >> req.video_status;
        }
            
        if(req.mask&0x04)
        {
            ds >> req.audio_status;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaStatusResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaStatusResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaFrameNotify& notify)
    {
        ds << notify.mask;
        
        if(notify.mask&0x01)
        {
            ds.writestring(notify.session_id.c_str());
        }
        
        if(notify.mask&0x02)
        {
            ds << notify.frame_type;
            ds << notify.frame_av_seq;
            ds << notify.frame_seq;
            ds << notify.frame_base_time;
            ds << notify.frame_ts;
            ds << notify.frame_size;
        }
            
        if(notify.mask&0x04)
        {
            ds << notify.crc32_hash;
        }
        
        if(notify.mask&0x08)
        {
            ds << notify.offset;
            ds << notify.data_size;
            for(uint32 i=0; i<notify.data_size; ++i)
            {
                ds << notify.datas[i];
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaFrameNotify& notify)
    {
        ds >> notify.mask;
        
        if(notify.mask&0x01)
        {
            notify.session_id = ds.readstring();
        }
        
        if(notify.mask&0x02)
        {
            ds >> notify.frame_type;
            ds >> notify.frame_av_seq;
            ds >> notify.frame_seq;
            ds >> notify.frame_base_time;
            ds >> notify.frame_ts;
            ds >> notify.frame_size;
        }
            
        if(notify.mask&0x04)
        {
            ds >> notify.crc32_hash;
        }
        
        if(notify.mask&0x08)
        {
            ds >> notify.offset;
            ds >> notify.data_size;
            for(uint32 i=0; i<notify.data_size; ++i)
            {
                uint8 data;
                ds >> data;
                notify.datas.push_back(data);
            }
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaEosNotify& notify)
    {
        ds << notify.mask;
        if(notify.mask&0x01)
        {
            ds.writestring(notify.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaEosNotify& notify)
    {
        ds >> notify.mask;
        if(notify.mask&0x01)
        {
            notify.session_id = ds.readstring();
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaCloseReq& req)
    {
        ds << req.mask;        
        if(req.mask&0x01)
        {
            ds.writestring(req.session_id.c_str());
        }     
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaCloseReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            req.session_id = ds.readstring();
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, StreamMediaCloseResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.session_id.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, StreamMediaCloseResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.session_id = ds.readstring();
        }
        return ds;
    }
}



