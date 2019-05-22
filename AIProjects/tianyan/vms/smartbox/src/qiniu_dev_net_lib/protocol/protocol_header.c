#define __PROTOCOL_HEADER_C__

#include "protocol_header.h"
#include "vos_bit_t.h"

int Pack_MsgHeader(OUT void* buf, IN vos_size_t buf_len, IN MsgHeader* msg_head)
{
    char* start_pos = NULL;
    char* pos = NULL;

    do
    {
        start_pos = (char*)buf;
        pos = start_pos;

        S2Bytes(pos, msg_head->msg_size);
        S4Bytes(pos, msg_head->msg_id);
        S4Bytes(pos, msg_head->msg_type);
        S4Bytes(pos, msg_head->msg_seq);

        return (pos-start_pos);
    }while (0);
    return -1;
}

int Pack_DevStreamInfo(OUT void* buf, IN vos_size_t buf_len, IN DevStreamInfo* stream_info)
{
    char* start_pos = NULL;
    char* pos = NULL;

    do
    {
        start_pos = (char*)buf;
        pos = start_pos;

        S1Bytes(pos, stream_info->stream_id);
        S4Bytes(pos, stream_info->video_height);
        S4Bytes(pos, stream_info->video_width);
        S1Bytes(pos, stream_info->video_codec.codec_fmt);

        return (pos-start_pos);
    }while (0);
    return -1;
}

int Pack_DevChannelInfo(OUT void* buf, IN vos_size_t buf_len, IN DevChannelInfo* channel_info)
{
    char* start_pos = NULL;
    char* pos = NULL;

    do
    {
        start_pos = (char*)buf;
        pos = start_pos;

        S2Bytes(pos, channel_info->channel_id);
        S2Bytes(pos, channel_info->channel_type);
        S1Bytes(pos, channel_info->channel_status);
        S1Bytes(pos, channel_info->has_ptz);
        S1Bytes(pos, channel_info->stream_num);
        {
            int i = 0;
            for( ; i < channel_info->stream_num; ++i )
            {
                int len = Pack_DevStreamInfo( pos, buf_len - (pos-start_pos), &channel_info->stream_list[i] );
                if( len < 0 )
                {
                    return len;
                }
                pos += len;
            }
        }

        S1Bytes(pos, channel_info->audio_codec.codec_fmt);
        S1Bytes(pos, channel_info->audio_codec.channel);
        S1Bytes(pos, channel_info->audio_codec.sample);
        S1Bytes(pos, channel_info->audio_codec.bitwidth);

        if( channel_info->audio_codec.sepc_size &&
            channel_info->audio_codec.sepc_size <= sizeof(channel_info->audio_codec.sepc_data) )
        {
            S1Bytes(pos, channel_info->audio_codec.sepc_size);
            SnBytes(pos, channel_info->audio_codec.sepc_data, channel_info->audio_codec.sepc_size);
        }
        else
        {
            char sepc_size = 0;
            S1Bytes(pos, sepc_size);
        }
        return (pos-start_pos);
    }while (0);
    return -1;
}

int Unpack_MsgHeader(IN void* head_buf, IN vos_size_t head_len, OUT MsgHeader* msg_head)
{
    do
    {
        char* pos = NULL;

        if ( !head_buf || !msg_head)
        {
            break;
        }

        if ( head_len < sizeof(MsgHeader ) )
        {
            break;
        }

        pos = (char*)head_buf;;

        R2Bytes(pos, msg_head->msg_size);
        R4Bytes(pos, msg_head->msg_id);
        R4Bytes(pos, msg_head->msg_type);
        R4Bytes(pos, msg_head->msg_seq);

        return 0;
    }while (0);
    return -1;
}
