#define __PROTOCOL_DEVICE_C__

#include "protocol_device.h"
#include "vos_bit_t.h"

int Pack_MsgDeviceLoginReq(OUT void* buf, IN vos_size_t buf_len, IN DeviceLoginReq* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->device_id);
            SnString(body_pos, msg->version);
            S1Bytes(body_pos, msg->dev_type);
            S2Bytes(body_pos, msg->channel_num);
            {
                int i = 0;
                for(; i<msg->channel_num; ++i)
                {
                    int len = Pack_DevChannelInfo(body_pos, start_pos+buf_len - body_pos, &msg->channels[i]);
                    if( len < 0 )
                    {
                        return len;
                    }
                    
                    body_pos += len;
                }
            }
            S2Bytes(body_pos, msg->token.token_bin_length);
            SnBytes(body_pos, msg->token.token_bin, msg->token.token_bin_length);
        }

        if( 0x02 & msg->mask )
        {
            SnString(body_pos, msg->private_addr.ip);
            S2Bytes(body_pos, msg->private_addr.port);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgDeviceAbilityReportReq(OUT char* buf, IN vos_size_t buf_len, IN DeviceAbilityReportReq* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            S1Bytes(body_pos, msg->media_trans_type);
            S1Bytes(body_pos, msg->max_live_streams_per_ch);
            S1Bytes(body_pos, msg->max_playback_streams_per_ch);
            S1Bytes(body_pos, msg->max_playback_streams);
        }

        if( 0x02 & msg->mask )
        {
            S4Bytes(body_pos, msg->disc_size);
            S4Bytes(body_pos, msg->disc_free_size);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;

}

int Pack_MsgDeviceStatusReportReq(OUT char* buf, IN vos_size_t buf_len, IN DeviceStatusReportReq* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            int i = 0;
            S1Bytes(body_pos, msg->channel_num);
            for(; i<msg->channel_num; ++i)
            {
                int len = Pack_DevChannelInfo( body_pos, start_pos+buf_len-body_pos, &msg->channels[i] );
                if( len < 0 )
                {
                    return len;
                }
                body_pos += len;
            }
        }

        if( 0x02 & msg->mask )
        {
            int i = 0;
            S2Bytes(body_pos, msg->media_session_num);
            for(; i<msg->media_session_num; ++i)
            {
                SnString(body_pos, msg->media_sessions[i].session_id);
                S2Bytes(body_pos, msg->media_sessions[i].session_type);
                S1Bytes(body_pos, msg->media_sessions[i].session_media);
                S1Bytes(body_pos, msg->media_sessions[i].session_status);
                SnString(body_pos, msg->media_sessions[i].device_id);
                S2Bytes(body_pos, msg->media_sessions[i].channel_id);
                S1Bytes(body_pos, msg->media_sessions[i].stream_id);
                SnString(body_pos, msg->media_sessions[i].stream_addr.ip);
                S2Bytes(body_pos, msg->media_sessions[i].stream_addr.port);
            }
        }

        if ( 0x04 & msg->mask )
        {
            S2Bytes(body_pos, msg->sdcard_status);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgDeviceAlarmReportReq(OUT char* buf, IN vos_size_t buf_len, IN DeviceAlarmReportReq* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->device_id);
            S2Bytes(body_pos, msg->channel_id);
            S4Bytes(body_pos, msg->alarm_type);
            S1Bytes(body_pos, msg->alarm_datas);
        }

        if( 0x02 & msg->mask )
        {
            S4Bytes(body_pos, msg->alarm_data_size);
            SnBytes(body_pos, msg->alarm_datas, msg->alarm_data_size);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgDeviceMediaOpenResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceMediaOpenResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);

        if ( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->device_id);
            S2Bytes(body_pos, msg->channel_id);
            S1Bytes(body_pos, msg->stream_id);
            SnString(body_pos, msg->session_id);
            SnString(body_pos, msg->stream_server.ip);
            S2Bytes(body_pos, msg->stream_server.port);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgDeviceMediaCloseResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceMediaCloseResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgDeviceSnapResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceSnapResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);

        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->device_id);
            S2Bytes(body_pos, msg->channel_id);
            SnString(body_pos, msg->pic_fmt);
            S4Bytes(body_pos, msg->pic_size);
        }

        if( 0x02 & msg->mask )
        {
            S4Bytes(body_pos, msg->offset);
            S4Bytes(body_pos, msg->data_size);
            SnBytes(body_pos, msg->datas, msg->data_size);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgDeviceCtrlResp(OUT char* buf, IN vos_size_t buf_len, IN DeviceCtrlResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaConnectReq(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaConnectReq* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            S2Bytes(body_pos, msg->session_type);
            SnString(body_pos, msg->session_id);
            S1Bytes(body_pos, msg->session_media);
            SnString(body_pos, msg->endpoint_name);
            S2Bytes(body_pos, msg->endpoint_type);
            SnString(body_pos, msg->device_id);
            S2Bytes(body_pos, msg->channel_id);
            S1Bytes(body_pos, msg->stream_id);
            S2Bytes(body_pos, msg->token.token_bin_length);
            SnBytes(body_pos, msg->token.token_bin, msg->token.token_bin_length);
        }

        if( 0x02 & msg->mask )
        {
            S1Bytes(body_pos, msg->video_direct);
            S1Bytes(body_pos, msg->video_codec.codec_fmt);
        }

        if( 0x04 & msg->mask )
        {
            S1Bytes(body_pos, msg->audio_direct);
            S1Bytes(body_pos, msg->audio_codec.codec_fmt);
            S1Bytes(body_pos, msg->audio_codec.channel);
            S1Bytes(body_pos, msg->audio_codec.sample);
            S1Bytes(body_pos, msg->audio_codec.bitwidth);

            if( msg->audio_codec.sepc_size &&
                msg->audio_codec.sepc_size <= sizeof(msg->audio_codec.sepc_data) )
            {
                S1Bytes(body_pos, msg->audio_codec.sepc_size);
                SnBytes(body_pos, msg->audio_codec.sepc_data, msg->audio_codec.sepc_size);
            }
            else
            {
                char sepc_size = 0;
                S1Bytes(body_pos, sepc_size);
            }
        }

        if ( 0x08 & msg->mask )
        {
            S4Bytes(body_pos, msg->begin_time);
            S4Bytes(body_pos, msg->end_time);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaDisconnectReq(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaDisconnectReq* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaStatusReq(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaStatusReq* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        if( 0x02 & msg->mask )
        {
            S1Bytes(body_pos, msg->video_status);
        }

        if( 0x04 & msg->mask )
        {
            S1Bytes(body_pos, msg->audio_status);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaPlayResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaPlayResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);
        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaPauseResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaPauseResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);
        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaCmdResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaCmdResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);
        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaCloseResp(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaCloseResp* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);
        S4Bytes(body_pos, msg->resp_code);
        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaFrameNotify(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaFrameNotify* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        if( 0x02 & msg->mask )
        {
            S1Bytes(body_pos, msg->frame_type);
            S4Bytes(body_pos, msg->frame_av_seq);
            S4Bytes(body_pos, msg->frame_seq);
            S4Bytes(body_pos, msg->frame_base_time);
            S4Bytes(body_pos, msg->frame_ts);
            S4Bytes(body_pos, msg->frame_size);
        }

        if( 0x04 & msg->mask )
        {
            S4Bytes(body_pos, msg->crc32_hash);
        }

        if ( 0x08 & msg->mask )
        {
            S4Bytes(body_pos, msg->offset);
            S4Bytes(body_pos, msg->data_size);
            SnBytes(body_pos, msg->datas, msg->data_size);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Pack_MsgStreamMediaEosNotify(OUT char* buf, IN vos_size_t buf_len, IN StreamMediaEosNotify* msg)
{
    char* start_pos = NULL;
    char* body_pos = NULL;

    do
    {
        start_pos = (char*)buf;
        body_pos = start_pos;

        S4Bytes(body_pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            SnString(body_pos, msg->session_id);
        }

        return (body_pos-start_pos);

    }while (0);

    return -1;
}

int Unpack_MsgDeviceLoginResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceLoginResp* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);
        R4Bytes(pos, msg->resp_code);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->public_addr.ip);
            R2Bytes(pos, msg->public_addr.port);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgDeviceAbilityReportResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceAbilityReportResp* msg)
{
    do
    {
        char* pos = NULL;
        pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);
        R4Bytes(pos, msg->resp_code);

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgDeviceStatusReportResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceStatusReportResp* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);
        R4Bytes(pos, msg->resp_code);

        if ( 0x01 & msg->mask )
        {
            R2Bytes(pos, msg->expected_cycle);
        }
        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgDeviceAlarmReportResp(IN void* msg_buf, IN vos_size_t msg_len, OUT DeviceAlarmReportResp* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);
        R4Bytes(pos, msg->resp_code);

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgDeviceMediaOpenReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceMediaOpenReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            RnString(pos, msg->device_id);
            R2Bytes(pos, msg->channel_id);
            R1Bytes(pos, msg->stream_id);
            R2Bytes(pos, msg->session_type);
            RnString(pos, msg->session_id);
            R1Bytes(pos, msg->session_media);
        }

        if( 0x02 & msg->mask )
        {
            R1Bytes(pos, msg->video_codec.codec_fmt);
        }

        if( 0x04 & msg->mask )
        {
            R1Bytes(pos, msg->audio_codec.codec_fmt);
            R1Bytes(pos, msg->audio_codec.channel);
            R1Bytes(pos, msg->audio_codec.sample);
            R1Bytes(pos, msg->audio_codec.bitwidth);
            R1Bytes(pos, msg->audio_codec.sepc_size);
            if( msg->audio_codec.sepc_size )
            {
                if( msg->audio_codec.sepc_size > sizeof(msg->audio_codec.sepc_data) )
                {
                    return -1;
                }
                RnBytes(pos, msg->audio_codec.sepc_data, msg->audio_codec.sepc_size);
            }
        }

        if( 0x08 & msg->mask )
        {
            R1Bytes(pos, msg->transport_type);
        }

        if ( 0x10 & msg->mask )
        {
            R4Bytes(pos, msg->begin_time);
            R4Bytes(pos, msg->end_time);
        }

        if ( 0x20 & msg->mask )
        {
            int i = 0;
            R2Bytes(pos, msg->stream_token.token_bin_length);
            RnBytes(pos, msg->stream_token.token_bin, msg->stream_token.token_bin_length);
            R2Bytes(pos, msg->stream_num);
            for(; i<msg->stream_num; ++i)
            {
                RnString(pos, msg->stream_servers[i].ip);
                R2Bytes(pos, msg->stream_servers[i].port);
            }
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgDeviceMediaCloseReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceMediaCloseReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            RnString(pos, msg->device_id);
            R2Bytes(pos, msg->channel_id);
            R1Bytes(pos, msg->stream_id);
            RnString(pos, msg->session_id);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgDeviceSnapReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceSnapReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            RnString(pos, msg->device_id);
            R2Bytes(pos, msg->channel_id);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgDeviceCtrlReq(IN char* msg_buf, IN vos_size_t msg_len, OUT DeviceCtrlReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if( 0x01 & msg->mask )
        {
            RnString(pos, msg->device_id);
            R2Bytes(pos, msg->channel_id);
            R2Bytes(pos, msg->cmd_type);
        }

        if ( 0x02 & msg->mask )
        {
            R2Bytes(pos, msg->cmd_data_size);
            RnBytes(pos, msg->cmd_datas, msg->cmd_data_size);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgStreamMediaConnectResp(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaConnectResp* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);
        R4Bytes(pos, msg->resp_code);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->session_id);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgStreamMediaDisconnectResp(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaDisconnectResp* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);
        R4Bytes(pos, msg->resp_code);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->session_id);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgStreamMediaStatusResp(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaStatusResp* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);
        R4Bytes(pos, msg->resp_code);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->session_id);
        }

        return 0;
    }while (0);
    return -1;    
}

int Unpack_MsgStreamMediaPlayReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaPlayReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->session_id);
        }

        if ( 0x02 & msg->mask )
        {
            R4Bytes(pos, msg->begin_time);
            R4Bytes(pos, msg->end_time);
        }

        if ( 0x04 & msg->mask )
        {
            R1Bytes(pos, msg->speed);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgStreamMediaPauseReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaPauseReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->session_id);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgStreamMediaCmdReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaCmdReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->session_id);
            R1Bytes(pos, msg->cmd_type);
        }

        if ( 0x02 & msg->mask )
        {
            R2Bytes(pos, msg->param_data_size);
            RnBytes(pos, msg->param_datas, msg->param_data_size);
        }

        return 0;
    }while (0);
    return -1;
}

int Unpack_MsgStreamMediaCloseReq(IN char* msg_buf, IN vos_size_t msg_len, OUT StreamMediaCloseReq* msg)
{
    do
    {
        char* pos = (char*)msg_buf;

        R4Bytes(pos, msg->mask);

        if ( 0x01 & msg->mask )
        {
            RnString(pos, msg->session_id);
        }

        return 0;
    }while (0);
    return -1;
}
