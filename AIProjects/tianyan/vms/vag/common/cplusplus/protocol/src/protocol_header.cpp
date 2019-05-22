#include <stdlib.h>
#include "protocol_header.h"
namespace protocol{

    CDataStream& operator<<(CDataStream &ds, MsgHeader &msg)
    {
        ds << msg.msg_size;
        ds << msg.msg_id;
        ds << msg.msg_type;
        ds << msg.msg_seq;
        return ds;
    }
    CDataStream& operator>>(CDataStream &ds, MsgHeader &msg)
    {
        ds >> msg.msg_size;
        ds >> msg.msg_id;
        ds >> msg.msg_type;
        ds >> msg.msg_seq;
        return ds;
    }
    
    CDataStream& operator<<( CDataStream& ds,token_t & token )
    {
        assert( token.token_bin_length >= 0 && token.token_bin_length <= 256 );
        ds << token.token_bin_length;
        ds.writedata( token.token_bin, token.token_bin_length );
        return ds;
    }

    CDataStream& operator>>( CDataStream& ds, token_t& token )
    {
        ds >> token.token_bin_length;
        if( token.token_bin_length > 0 && token.token_bin_length <= 256 )
        {
            ds.readdata( token.token_bin_length , token.token_bin );
        }
        else if(token.token_bin_length)
        {
            ds.good_bit(false);
        }
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, HostAddr& hi)
    {
        ds.writestring(hi.ip.c_str());
        ds << hi.port;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, HostAddr& hi)
    {
        hi.ip = ds.readstring();
        ds >> hi.port;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, HistoryRecordBlock& blk)
    {
        ds << blk.begin_time;
        ds << blk.end_time;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, HistoryRecordBlock& blk)
    {
        ds >> blk.begin_time;
        ds >> blk.end_time;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, SBandwidthInfo& bw)
    {
        ds << bw.upload_bw;
        ds << bw.download_bw;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, SBandwidthInfo& bw)
    {
        ds >> bw.upload_bw;
        ds >> bw.download_bw;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, STimeVal& tv)
    {
        ds << tv.tv_sec;
        ds << tv.tv_usec;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, STimeVal& tv)
    {
        ds >> tv.tv_sec;
        ds >> tv.tv_usec;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, ChannelStatus& status)
    {
        ds << status.channel_id;
        ds << status.status;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, ChannelStatus& status)
    {
        ds >> status.channel_id;
        ds >> status.status;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, AudioCodecInfo& data)
    {
        ds << data.codec_fmt;
        ds << data.channel;
        ds << data.sample;
        ds << data.bitwidth;
        ds << data.sepc_size;
        vector<uint8>::iterator it = data.sepc_data.begin();
        for( ; it != data.sepc_data.end(); ++it )
        {
            ds << *it;
        }
        return ds;
    }

    CDataStream& operator>>(CDataStream& ds, AudioCodecInfo& data)
    {
        ds >> data.codec_fmt;
        ds >> data.channel;
        ds >> data.sample;
        ds >> data.bitwidth;
        ds >> data.sepc_size;
        for( int i = 0; i < data.sepc_size; ++i )
        {
            uint8 a;
            ds >> a;
            data.sepc_data.push_back(a);
        }
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, VideoCodecInfo& data)
    {
        ds << data.codec_fmt;
        return ds;
    }

    CDataStream& operator>>(CDataStream& ds, VideoCodecInfo& data)
    {
        ds >> data.codec_fmt;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, DevAttribute& data)
    {
        ds << data.has_microphone;
        ds << data.has_hard_disk;
        ds << data.can_recv_audio;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DevAttribute& data)
    {
        ds >> data.has_microphone;
        ds >> data.has_hard_disk;
        ds >> data.can_recv_audio;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, DevOEMInfo& data)
    {
        ds << data.oem_id;
        ds.writestring(data.oem_name.c_str());
        ds.writestring(data.mac.c_str());
        ds.writestring(data.sn.c_str());
        ds.writestring(data.model.c_str());
        ds.writestring(data.factory.c_str());
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DevOEMInfo& data)
    {
        ds >> data.oem_id;
        data.oem_name = ds.readstring();
        data.mac = ds.readstring();
        data.sn = ds.readstring();
        data.model = ds.readstring();
        data.factory = ds.readstring();
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, DevStreamInfo& data)
    {
        ds << data.stream_id;
        ds << data.video_height;
        ds << data.video_width;
        ds << data.video_codec;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DevStreamInfo& data)
    {
        ds >> data.stream_id;
        ds >> data.video_height;
        ds >> data.video_width;
        ds >> data.video_codec;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, DevChannelInfo& data)
    {
        ds << data.channel_id;
        ds << data.channel_type;
        ds << data.channel_status;
        ds << data.has_ptz;
        ds << data.stream_num;

        vector<DevStreamInfo>::iterator it = data.stream_list.begin();
        for( ; it != data.stream_list.end(); ++it )
        {
            ds << *it;
        }

        ds << data.audio_codec;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, DevChannelInfo& data)
    {
        ds >> data.channel_id;
        ds >> data.channel_type;
        ds >> data.channel_status;
        ds >> data.has_ptz;
        ds >> data.stream_num;
        for( int i = 0; i < data.stream_num; ++i )
        {
            DevStreamInfo stream;
            ds >> stream;
            data.stream_list.push_back(stream);
        }

        ds >> data.audio_codec;
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, MsgTunnelReq& req)
    {
        ds << req.mask;
        if( req.mask & 0x01 )
        {
            ds << req.tunnel_type;
        }
        if( req.mask & 0x02 )
        {
            ds << req.req_data_size;
            for (int i = 0; i< req.req_data_size; ++i)
            {
                ds << req.req_datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, MsgTunnelReq& req)
    {
        ds >> req.mask;
        if( req.mask & 0x01 )
        {
            ds >> req.tunnel_type;
        }
        if( req.mask & 0x02 )
        {
            ds >> req.req_data_size;
            for (int i = 0; i< req.req_data_size; ++i)
            {
                uint8 data;
                ds >> data;
                req.req_datas.push_back(data);
            }
        }
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, MsgTunnelResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        if( resp.mask & 0x01 )
        {
            ds << resp.resp_data_size;
            for (int i = 0; i< resp.resp_data_size; ++i)
            {
                ds << resp.resp_datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, MsgTunnelResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        if( resp.mask & 0x01 )
        {
            ds >> resp.resp_data_size;
            for (int i = 0; i< resp.resp_data_size; ++i)
            {
                uint8 data;
                ds >> data;
                resp.resp_datas.push_back(data);
            }
        }
        return ds;
    }
}



