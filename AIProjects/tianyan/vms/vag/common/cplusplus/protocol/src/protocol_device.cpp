
#include "protocol_device.h"
namespace protocol{

    CDataStream& operator<<(CDataStream& ds, PtzCmdData& status)
    {
        ds << status.opt_type;
        ds << status.param1;
        ds << status.param2;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, PtzCmdData& status)
    {
        ds >> status.opt_type;
        ds >> status.param1;
        ds >> status.param2;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, DeviceMediaSessionStatus& status)
    {
        ds.writestring(status.session_id.c_str());
        ds << status.session_type;
        ds << status.session_media;
        ds << status.session_status;
        ds.writestring(status.device_id.c_str());
        ds << status.channel_id;
        ds << status.stream_id;
        ds << status.stream_addr;
        return ds;        
    }
    CDataStream& operator>>(CDataStream& ds, DeviceMediaSessionStatus& status)
    {
        status.session_id = ds.readstring();
        ds >> status.session_type;
        ds >> status.session_media;
        ds >> status.session_status;
        status.device_id = ds.readstring();
        ds >> status.channel_id;
        ds >> status.stream_id;
        ds >> status.stream_addr;
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceLoginReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds.writestring(req.version.c_str());
            ds << req.dev_type;
            ds << req.channel_num;
            for(uint32 i=0; i<req.channel_num;++i)
            {
                ds << req.channels[i];
            }
            ds << req.token;
        }
        
        if(req.mask&0x02)
        {
            ds << req.private_addr;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceLoginReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            req.version = ds.readstring();
            ds >> req.dev_type;
            ds >> req.channel_num;
            for(uint32 i=0; i<req.channel_num;++i)
            {
                DevChannelInfo channel;
                ds >> channel;
                req.channels.push_back(channel);
            }
            ds >> req.token;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.private_addr;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceLoginResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds << resp.public_addr;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceLoginResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds >> resp.public_addr;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceAbilityReportReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds << req.media_trans_type;
            ds << req.max_live_streams_per_ch;
            ds << req.max_playback_streams_per_ch;
            ds << req.max_playback_streams;
        }
        
        if(req.mask&0x02)
        {
            ds << req.disc_size;
            ds << req.disc_free_size;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceAbilityReportReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            ds >> req.media_trans_type;
            ds >> req.max_live_streams_per_ch;
            ds >> req.max_playback_streams_per_ch;
            ds >> req.max_playback_streams;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.disc_size;
            ds >> req.disc_free_size;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceAbilityReportResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceAbilityReportResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceStatusReportReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds << req.channel_num;
            for(uint32 i=0; i<req.channel_num;++i)
            {
                ds << req.channels[i];
            }
        }
        
        if(req.mask&0x02)
        {
            ds << req.media_session_num;
            for(uint32 i=0; i<req.media_session_num;++i)
            {
                ds << req.media_sessions[i];
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceStatusReportReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            ds >> req.channel_num;
            for(uint32 i=0; i<req.channel_num;++i)
            {
                DevChannelInfo channel;
                ds >> channel;
                req.channels.push_back(channel);
            }
        }
        
        if(req.mask&0x02)
        {
            ds >> req.media_session_num;
            for(uint32 i=0; i<req.media_session_num;++i)
            {
                DeviceMediaSessionStatus session;
                ds >> session;
                req.media_sessions.push_back(session);
            }
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceStatusReportResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds << resp.expected_cycle;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceStatusReportResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds >> resp.expected_cycle;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceAlarmReportReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.alarm_type;
            ds << req.alarm_status;
        }
        
        if(req.mask&0x02)
        {
            ds << req.alarm_data_size;
            for(uint32 i=0; i< req.alarm_data_size; ++i)
            {
                ds << req.alarm_datas[i];
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceAlarmReportReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.alarm_type;
            ds >> req.alarm_status;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.alarm_data_size;
            for(uint32 i=0; i< req.alarm_data_size; ++i)
            {
                uint8 data;
                ds >> data;
                req.alarm_datas.push_back(data);
            }
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceAlarmReportResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceAlarmReportResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DevicePictureUploadReq& req)
    {
        ds << req.mask;
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DevicePictureUploadReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds >> req.channel_id;
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DevicePictureUploadResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.upload_url.c_str());
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DevicePictureUploadResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.upload_url = ds.readstring();
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceSnapReq& req)
    {
        ds << req.mask;
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceSnapReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceSnapResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;

        if(resp.mask&0x01)
        {
            ds.writestring(resp.device_id.c_str());
            ds << resp.channel_id;
            ds.writestring(resp.pic_fmt.c_str());
            ds << resp.pic_size;
        }

        if(resp.mask&0x02)
        {
            ds << resp.offset;
            ds << resp.data_size;
            for(uint32 i=0;i<resp.data_size;++i)
            {
                ds << resp.datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceSnapResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;

        if(resp.mask&0x01)
        {
            resp.device_id = ds.readstring();
            ds >> resp.channel_id;
            resp.pic_fmt = ds.readstring();
            ds >> resp.pic_size;
        }

        if(resp.mask&0x02)
        {
            ds >> resp.offset;
            ds >> resp.data_size;
            for(uint32 i=0;i<resp.data_size;++i)
            {
                uint8 data;
                ds >> data;
                resp.datas.push_back(data);
            }
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceCtrlReq& req)
    {
        ds << req.mask;
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.cmd_type;
        }
        if(req.mask&0x02)
        {
            ds << req.cmd_data_size;
            for(uint32 i=0;i<req.cmd_data_size;++i)
            {
                ds << req.cmd_datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceCtrlReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.cmd_type;
        }
        if(req.mask&0x02)
        {
            ds >> req.cmd_data_size;
            for(uint32 i=0;i<req.cmd_data_size;++i)
            {
                uint8 data;
                ds >> data;
                req.cmd_datas.push_back(data);
            }
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceCtrlResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceCtrlResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceParamSetReq& req)
    {
        ds << req.mask;
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.param_type;
        }
        if(req.mask&0x02)
        {
            ds << req.param_data_size;
            for(uint32 i=0;i<req.param_data_size;++i)
            {
                ds << req.param_datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceParamSetReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.param_type;
        }
        if(req.mask&0x02)
        {
            ds >> req.param_data_size;
            for(uint32 i=0;i<req.param_data_size;++i)
            {
                uint8 data;
                ds >> data;
                req.param_datas.push_back(data);
            }
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceParamSetResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceParamSetResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceParamGetReq& req)
    {
        ds << req.mask;
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.param_type;
        }
        if(req.mask&0x02)
        {
            ds << req.param_data_size;
            for(uint32 i=0;i<req.param_data_size;++i)
            {
                ds << req.param_datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceParamGetReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.param_type;
        }
        if(req.mask&0x02)
        {
            ds >> req.param_data_size;
            for(uint32 i=0;i<req.param_data_size;++i)
            {
                uint8 data;
                ds >> data;
                req.param_datas.push_back(data);
            }
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceParamGetResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.device_id.c_str());
            ds << resp.channel_id;
            ds << resp.param_type;
        }
        if(resp.mask&0x02)
        {
            ds << resp.param_data_size;
            for(uint32 i=0;i<resp.param_data_size;++i)
            {
                ds << resp.param_datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceParamGetResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.device_id = ds.readstring();
            ds >> resp.channel_id;
            ds >> resp.param_type;
        }
        if(resp.mask&0x02)
        {
            ds >> resp.param_data_size;
            for(uint32 i=0;i<resp.param_data_size;++i)
            {
                uint8 data;
                ds >> data;
                resp.param_datas.push_back(data);
            }
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceRecordListQueryReq& req)
    {
        ds << req.mask;
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.stream_id;
            ds << req.begin_time;
            ds << req.end_time;
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceRecordListQueryReq& req)
    {
        ds >> req.mask;
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.stream_id;
            ds >> req.begin_time;
            ds >> req.end_time;
        }
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceRecordListQueryResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if(resp.mask&0x01)
        {
            ds.writestring(resp.device_id.c_str());
            ds << resp.channel_id;
            ds << resp.stream_id;
            ds << resp.block_total_num;
            ds << resp.block_seq;
            ds << resp.block_num;
            for(uint32 i=0;i<resp.block_num;++i)
            {
                ds << resp.record_blocks[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceRecordListQueryResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if(resp.mask&0x01)
        {
            resp.device_id = ds.readstring();
            ds >> resp.channel_id;
            ds >> resp.stream_id;
            ds >> resp.block_total_num;
            ds >> resp.block_seq;
            ds >> resp.block_num;
            for(uint32 i=0;i<resp.block_num;++i)
            {
                HistoryRecordBlock block;
                ds >> block;
                resp.record_blocks.push_back(block);
            }
        }
        return ds;
    }
    
    
    CDataStream& operator<<(CDataStream& ds, DeviceMediaOpenReq& req)
    {
        ds << req.mask;
        
        if(req.mask&0x01)
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_id;
            ds << req.stream_id;
            ds << req.session_type;
            ds.writestring(req.session_id.c_str());
            ds << req.session_media;
        }
        
        if(req.mask&0x02)
        {
            ds << req.video_codec;
        }
        
        if(req.mask&0x04)
        {
            ds << req.audio_codec;
        }
        
        if(req.mask&0x08)
        {
            ds << req.transport_type;
        }
        
        if(req.mask&0x10)
        {
            ds << req.begin_time;
            ds << req.end_time;
        }
        
        if(req.mask&0x20)
        {
            ds << req.stream_token;
            ds << req.stream_num;
            for(uint32 i=0; i<req.stream_num;++i)
            {
                ds << req.stream_servers[i];
            }
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceMediaOpenReq& req)
    {
        ds >> req.mask;
        
        if(req.mask&0x01)
        {
            req.device_id = ds.readstring();
            ds >> req.channel_id;
            ds >> req.stream_id;
            ds >> req.session_type;
            req.session_id = ds.readstring();
            ds >> req.session_media;
        }
        
        if(req.mask&0x02)
        {
            ds >> req.video_codec;
        }
        
        if(req.mask&0x04)
        {
            ds >> req.audio_codec;
        }
        
        if(req.mask&0x08)
        {
            ds >> req.transport_type;
        }
        
        if(req.mask&0x10)
        {
            ds >> req.begin_time;
            ds >> req.end_time;
        }
        
        if(req.mask&0x20)
        {
            ds >> req.stream_num;
            for(uint32 i=0; i<req.stream_num;++i)
            {
                HostAddr host;
                ds >> host;
                req.stream_servers.push_back(host);
            }
            ds >> req.stream_token;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceMediaOpenResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        
        if(resp.mask&0x01)
        {
            ds.writestring(resp.device_id.c_str());
            ds << resp.channel_id;
            ds << resp.stream_id;
            ds.writestring(resp.session_id.c_str());
            ds << resp.stream_server;
        }
        
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceMediaOpenResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        
        if(resp.mask&0x01)
        {
            resp.device_id = ds.readstring();
            ds >> resp.channel_id;
            ds >> resp.stream_id;
            resp.session_id = ds.readstring();
            ds >> resp.stream_server;
        }
        
        return ds;
    }
    
    CDataStream& operator<<(CDataStream& ds, DeviceMediaCloseReq& req)
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
    CDataStream& operator>>(CDataStream& ds, DeviceMediaCloseReq& req)
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
    
    CDataStream& operator<<(CDataStream& ds, DeviceMediaCloseResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DeviceMediaCloseResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        return ds;
    }
}



