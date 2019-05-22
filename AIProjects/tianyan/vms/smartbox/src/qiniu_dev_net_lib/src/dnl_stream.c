#define __DNL_STREAM_C__

#include "dnl_stream.h"
#include "dnl_transport.h"
#include "dnl_util.h"
#include "dnl_log.h"
#include "dnl_dev.h"
#include "dnl_ctrl.h"
#include "dnl_entry.h"
#include "vos_socket.h"
#include "cdiffie_hell_man.h"
#include "protocol_device.h"

//流会话程序入口
static int stream_session_conttbl_proc(media_session_t* session);

//流会话通信打开处理
static int stream_session_transport_open(media_session_t* session);

//流会话状态变更处理
static void stream_session_status_change(media_session_t* session, stream_session_status_e status);

//流会话状态上报
static int stream_session_status_report(media_session_t *session);

//流会话状态上报
static int stream_session_frame_report(media_session_t *session);

//其他
static int stream_media_frame_enque(media_session_t* session, media_frame_data_t *frame_pkg);

//流会话状态控制表
static dnl_conttbl_t stream_session_conttbl_tbl[] = 
{
    /*     function                     OK    NG   CONT	*/
    {&stream_session_on_tp_open,        1,    4,    0 },
    {&stream_session_on_exchange,       2,    4,    0 },
    {&stream_session_on_media_connect,  3,	  4,    0 },
    {&stream_session_on_normal_run,     3,    4,    0 },
    {&stream_session_on_close,          0,	  4,	0 }
};

int dnl_stream_init(void)
{
    memset(&g_DnlStreamEp, 0x0, sizeof(g_DnlStreamEp));

    g_DnlStreamEp.running = TRUE;
    
    g_DnlStreamEp.live_session_num = 0;

    vos_list_init( &g_DnlStreamEp.live_session_list );

    return 0;
}

int dnl_stream_final(void)
{
    g_DnlStreamEp.running = FALSE;

    {
        media_session_t* p = g_DnlStreamEp.live_session_list.next;
        while( p != &g_DnlStreamEp.live_session_list )
        {
            media_session_t* next = p->next;

            vos_list_erase(p);
            VOS_FREE_T(p);

            p = next;
        }

        g_DnlStreamEp.live_session_num = 0;
    }

    return 0;
}

int stream_1s_timer(void)
{
    return 0;
}

vos_thread_ret_t stream_run_proc(vos_thread_arg_t arg)
{
    int i = 0;
    media_session_t* p;

    if ( !g_DnlStreamEp.running )
    {
        return -1;
    }

    p = g_DnlStreamEp.live_session_list.next;
    while( p != &g_DnlStreamEp.live_session_list )
    {
        if( !p->running )
        {
            media_session_t* next = p->next;
            
            vos_list_erase(p);
            if( p->tp.tx.buf )
            {
                VOS_FREE_T(p->tp.tx.buf);
            }

            if( p->tp.rx.buf )
            {
                VOS_FREE_T(p->tp.rx.buf);
            }

            VOS_FREE_T(p);

            p = next;
            continue;
        }

        stream_session_conttbl_proc( p );

        p = p->next;
    }

    return 0;
}

int stream_media_event_report(media_session_t *session, Dev_Stream_Frame_t *stream)
{
    int ret, pkg_cnt, i;
    vos_uint32_t offset = 0, last_pkg_data_len=0;
    media_frame_data_t frame_pkg;

    if ( stream->frame_type == EN_FRM_TYPE_I )	//I帧
    {
        frame_pkg.frm_type = FRAME_TYPE_I;
    }
    else if (stream->frame_type == EN_FRM_TYPE_P)	//P帧
    {
        frame_pkg.frm_type = FRAME_TYPE_P;
    }
    else
    {
        frame_pkg.frm_type = FRAME_TYPE_AU;
    }

    frame_pkg.frm_id = stream->frame_id;
    frame_pkg.frm_av_id = stream->frame_av_id;
    frame_pkg.frm_ts = stream->frame_ts;
    frame_pkg.frm_size = stream->frame_size;

    /*
    DNL_TRACE_LOG(
        "stream_media_event_report, frm_type=%d, frm_id=%d, frm_av_id=%d, frm_ts=%d, frm_size=%d\n", 
        frame_pkg.frm_type,
        frame_pkg.frm_id,
        frame_pkg.frm_av_id,
        frame_pkg.frm_ts,
        frame_pkg.frm_size
        );*/

    pkg_cnt = frame_pkg.frm_size / FRAME_SNED_SIZE;
    for( i = 0; i < pkg_cnt; i++)
    {
        frame_pkg.frm_offset = stream->frame_offset + offset;
        frame_pkg.frm_data_size = FRAME_SNED_SIZE;
        memcpy(frame_pkg.frm_data, stream->pdata+offset, frame_pkg.frm_data_size);
        ret = stream_media_frame_enque(session, &frame_pkg);
        if(ret <0)
        {
            printf("stream enque failed!\n");
            return ret;
        }

        offset += FRAME_SNED_SIZE;
    }

    last_pkg_data_len = frame_pkg.frm_size % FRAME_SNED_SIZE;
    if(last_pkg_data_len)
    {
        frame_pkg.frm_offset = stream->frame_offset + offset;
        frame_pkg.frm_data_size = last_pkg_data_len;
        memcpy(frame_pkg.frm_data, stream->pdata+offset, frame_pkg.frm_data_size);
        ret = stream_media_frame_enque(session, &frame_pkg);
        if(ret <0)
        {
            printf("stream enque failed!\n");
            return ret;
        }
    }

#ifdef DEBUG_DUMP
    if(session->pf_stream_info)
    {
        char stream_info[256];
        memset(stream_info, 0x0, sizeof(stream_info));
        sprintf( stream_info, "frm_ts:%llu, frm_size:%u, frm_type:%u,frm_seq:%u\n",
            stream->frame_ts, stream->frame_size, (unsigned int)stream->frame_type, stream->frame_id );
        fwrite(stream_info, 1, strlen(stream_info), session->pf_stream_info);
    }

    if(session->pf_stream_data)
    {
        fwrite(stream->pdata, 1, stream->frame_size, session->pf_stream_data);
    }
#endif

    return 0;
}

int stream_media_session_open(media_desc_t desc)
{
    int ret = 0;

    do 
    {
        unsigned int i = 0;

        if( !g_DnlStreamEp.running )
        {
            ret = -1;
            break;
        }

        if( desc.session_type == MEDIA_SESSION_TYPE_LIVE )
        {
            media_session_t *session = stream_get_media_session( MEDIA_SESSION_TYPE_LIVE, desc.channel_id, desc.stream_id );

            if( session )
            {
                ret = -2;
                break;
            }

            session = VOS_MALLOC_T(media_session_t);
            if( !session )
            {
                ret = -3;
                break;
            }
            memset( session, 0x0, sizeof(media_session_t) );

            strncpy(session->session_id, desc.session_id, MAX_MEDIA_SESSION_ID_LEN);
            session->session_type = MEDIA_SESSION_TYPE_LIVE;
            session->channel_id = desc.channel_id;
            session->stream_id = desc.stream_id;
            session->is_audio_open = desc.is_audio_open;
            session->is_video_open = desc.is_video_open;

            session->media_frame.last_frm_ts = 0;
            session->media_frame.last_adjust_tv.sec = 0;
            session->media_frame.last_adjust_tv.usec = 0;
            session->media_frame.last_frame_offset = 0;
            session->media_frame.media_que.frm_nums = 0;
            session->media_frame.media_que.rd_idx = 0;
            session->media_frame.media_que.rw_idx = 0;

            session->start_time = vos_get_system_tick();
            session->report_cycle = 0;
            session->last_report_time = 0;
            session->load = 0;

            session->status = en_stream_session_on_init;
            session->is_pause = true;

            session->cont_seq = 0;
            session->sub_seq = 0;
            session->sub_start_time = 0;

            session->cur_stream_svr_idx = 0;
            session->stream_svr_list_len = desc.stream_svr_list_len;
            for(i=0; i<session->stream_svr_list_len;i++)
            {
                memcpy(&session->stream_svr_list[i], &desc.stream_svr_list[i], sizeof(addr_info));
            }

            session->req_seq = 0;
            session->retry_cnt = 0;
            session->running = TRUE;

#ifdef DEBUG_DUMP
            {
                char file_info[64]={0};
                char file_data[64]={0};
                int len = snprintf(file_info, sizeof(file_info)-1, "dnl_%s.dat", session->session_id);
                if(len > 0)
                {
                    file_info[len]='\0';
                    session->pf_stream_info = fopen(file_info, "wb");
                }
                
                len = snprintf(file_data, sizeof(file_data)-1, "dnl_%s.264", session->session_id);
                if(len > 0)
                {
                    file_data[len]='\0';
                    session->pf_stream_data = fopen(file_data, "wb");
                }
            }
#endif

            g_DnlStreamEp.live_session_num ++;
            vos_list_push_back( &g_DnlStreamEp.live_session_list, session );

            ret = 0;
        }

    } while (0);


    DNL_DEBUG_LOG(
        "media session open, session_id=%s, session_type=%d, ch_no=%d, rate=%d, ret=%d.\n", 
        desc.session_id,
        desc.session_type,
        desc.channel_id,
        desc.stream_id,
        ret );

    return ret;
}

int stream_media_session_close(char* session_id)
{
    media_session_t *session = stream_get_media_session_by_id(session_id);
    if( !session )
    {
        return 0;
    }

    session->running = FALSE;

#ifdef DEBUG_DUMP
    if( session->pf_stream_info)
    {
        fclose(session->pf_stream_info);
        session->pf_stream_info = NULL;
    }

    if( session->pf_stream_data )
    {
        fclose(session->pf_stream_data);
        session->pf_stream_data = NULL;
    }
#endif

    DNL_DEBUG_LOG(
        "media session close, session_id=%s, session_type=%d, ch_no=%d, rate=%d.\n",
        session->session_id,
        session->session_type,
        session->channel_id,
        session->stream_id );

    return 0;
}

media_session_t* stream_get_media_session(vos_uint16_t session_type, vos_int16_t ch_id, vos_uint8_t stream_id)
{
    if ( session_type==MEDIA_SESSION_TYPE_LIVE )
    {
        media_session_t* p = g_DnlStreamEp.live_session_list.next;
        for( ; p != &g_DnlStreamEp.live_session_list; p=p->next )
        {
            if ( p->running 
                && p->channel_id == ch_id 
                && p->stream_id == stream_id )
            {
                return p;
            }
        }
    }

    return NULL;
}

media_session_t* stream_get_media_session_by_id(char* session_id)
{
    media_session_t* p = g_DnlStreamEp.live_session_list.next;
    for( ; p != &g_DnlStreamEp.live_session_list; p=p->next )
    {
        if ( p->running 
            && strcmp(session_id, p->session_id) == 0 )
        {
            return p;
        }
    }

    return NULL;
}

media_session_t* stream_get_media_session_by_tp(dnl_transport_data_t* tp)
{
    media_session_t* p = g_DnlStreamEp.live_session_list.next;
    for( ; p != &g_DnlStreamEp.live_session_list; p=p->next )
    {
        if ( p->running 
            && (&p->tp == tp) )
        {
            return p;
        }
    }

    return NULL;
}

vos_bool_t stream_media_session_is_open(char* session_id)
{
    media_session_t* session = stream_get_media_session_by_id(session_id);
    if ( !session || !session->running )
    {
        return false;
    }

    if ( session->status != en_stream_session_on_play 
        && session->status != en_stream_session_on_pause )
    {
        return false;
    }

    return true;
}

int stream_build_msg_media_connect_req(media_session_t *session, void *msg_buf, vos_size_t buf_len)
{
    vos_size_t msg_size=0, body_size = 0;
    char *head_pos, *body_pos;

    if ( !session || !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    {
        StreamMediaConnectReq req;
        memset(&req, 0x0, sizeof(req));
        //0x01
        req.mask = 0x01;
        req.session_type = session->session_type;
        strncpy(req.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);
        if(session->is_video_open)
        {
            req.session_media |= 0x01;
        }
        if(session->is_audio_open)
        {
            req.session_media |= 0x02;
        }
        strncpy(req.endpoint_name, g_DnlDevInfo.dev_id, MAX_DEV_ID_LEN);
        req.endpoint_type = 1;
        strncpy(req.device_id, g_DnlDevInfo.dev_id, MAX_DEV_ID_LEN);
        req.channel_id = session->channel_id;
        req.stream_id = session->stream_id;
        memcpy(&req.token, &(session->token), sizeof(token_t));

        {
            vos_mutex_lock(g_DnlDevInfoMutex);
            //0x02
            req.mask |= 0x02;
            req.video_direct = MEDIA_DIR_SEND_ONLY;
            req.video_codec.codec_fmt= g_DnlDevInfo.channel_list[session->channel_id-1].stream_list[session->stream_id].video_codec.codec_fmt;

            //0x04
            req.mask |= 0x04;
            req.audio_direct = MEDIA_DIR_SEND_RECV;
            memcpy(&req.audio_codec, &g_DnlDevInfo.channel_list[session->channel_id-1].adudo_codec, sizeof(Dev_Audio_Codec_Info_t));
            vos_mutex_unlock(g_DnlDevInfoMutex);
        }
        
        //0x08
        req.mask |= 0x08;
        req.begin_time = session->begin_time;
        req.end_time = session->end_time;

        DNL_TRACE_LOG(
            "stream_build_msg_media_connect_req, session_type=%u, session_id=%s, session_media=%u, device_id=%s, channel_id=%u, stream_id=%u.\n", 
            req.session_type,
            req.session_id,
            req.session_media,
            req.device_id,
            req.channel_id,
            req.stream_id
            );

        body_size = Pack_MsgStreamMediaConnectReq(body_pos, buf_len-sizeof(MsgHeader), &req);
        if ( body_size<0 )
        {
            return -2;
        }
    }

    msg_size = body_size + sizeof(MsgHeader);

    {
        MsgHeader header;
        header.msg_size = msg_size;
        header.msg_id = MSG_ID_MEDIA_CONNECT;
        header.msg_type = MSG_TYPE_REQ;
        header.msg_seq = ++session->req_seq;

        if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
        {
            return -3;
        }
    }

    return msg_size;
}

int stream_build_msg_media_disconnect_req(media_session_t *session, void *msg_buf, vos_size_t buf_len)
{
    vos_size_t msg_size=0, body_size = 0;
    char *head_pos, *body_pos;

    if ( !session || !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    {
        StreamMediaDisconnectReq req;
        memset(&req, 0x0, sizeof(req));
        //0x01
        req.mask = 0x01;
        strncpy(req.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);

        DNL_DEBUG_LOG("stream_build_msg_media_disconnect_req, session_id=%s.\n", session->session_id );

        body_size = Pack_MsgStreamMediaDisconnectReq(body_pos, buf_len-sizeof(MsgHeader), &req);
        if ( body_size<0 )
        {
            return -2;
        }
    }

    msg_size = body_size + sizeof(MsgHeader);

    {
        MsgHeader header;
        header.msg_size = msg_size;
        header.msg_id = MSG_ID_MEDIA_DISCONNECT;
        header.msg_type = MSG_TYPE_REQ;
        header.msg_seq = ++session->req_seq;

        if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
        {
            return -3;
        }
    }

    return msg_size;
}

int stream_build_msg_media_status_report_req(media_session_t *session, void *msg_buf, vos_size_t buf_len)
{
    vos_size_t msg_size=0, body_size = 0;
    char *head_pos, *body_pos;

    if ( !session || !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    {
        StreamMediaStatusReq req;
        memset(&req, 0x0, sizeof(req));
        //0x01
        req.mask = 0x01;
        strncpy(req.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);
        //0x02
        req.mask |= 0x02;
        req.video_status = session->is_video_open?1:2;
        //0x04
        req.mask |= 0x04;
        req.audio_status = session->is_audio_open?1:2;

        DNL_DEBUG_LOG("stream_build_msg_media_status_report_req, session_id=%s, video_status=%u, audio_status=%u\n", 
            session->session_id, req.video_status, req.audio_status );

        body_size = Pack_MsgStreamMediaStatusReq(body_pos, buf_len-sizeof(MsgHeader), &req);
        if ( body_size<0 )
        {
            return -2;
        }
    }

    msg_size = body_size + sizeof(MsgHeader);

    {
        MsgHeader header;
        header.msg_size = msg_size;
        header.msg_id = MSG_ID_MEDIA_STATUS;
        header.msg_type = MSG_TYPE_REQ;
        header.msg_seq = ++session->req_seq;

        if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
        {
            return -3;
        }
    }

    return msg_size;
}

int stream_build_msg_media_eos_notify(media_session_t *session, void *msg_buf, vos_size_t buf_len)
{
    vos_size_t msg_size=0, body_size = 0;
    char *head_pos, *body_pos;

    if ( !session || !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    {
        StreamMediaEosNotify notify;
        memset(&notify, 0x0, sizeof(notify));
        //0x01
        notify.mask = 0x01;
        strncpy(notify.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);

        body_size = Pack_MsgStreamMediaEosNotify(body_pos, buf_len-sizeof(MsgHeader), &notify);
        if ( body_size<0 )
        {
            return -2;
        }
    }

    msg_size = body_size + sizeof(MsgHeader);

    {
        MsgHeader header;
        header.msg_size = msg_size;
        header.msg_id = MSG_ID_MEDIA_EOS;
        header.msg_type = MSG_TYPE_NOTIFY;
        header.msg_seq = ++session->req_seq;

        if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
        {
            return -3;
        }
    }

    return msg_size;
}

int stream_on_recv_msg(dnl_transport_data_t *tp, MsgHeader *header, void *msg_buf, vos_size_t msg_len)
{
    int ret = 0;
    do 
    {
        media_session_t *session = stream_get_media_session_by_tp(tp);
        if (!session || !session->running)
        {
            ret = -1;
            break;
        }

        if(header->msg_type == MSG_TYPE_REQ)
        {
            switch(header->msg_id)
            {
            case MSG_ID_MEDIA_PLAY:
                {
                    StreamMediaPlayReq req;
                    if ( Unpack_MsgStreamMediaPlayReq((char*)msg_buf, msg_len, &req) < 0 )
                    {
                        ret = -2;
                        break;
                    }
                    if ( stream_on_msg_media_play_req(session, header, &req) < 0 )
                    {
                        ret = -3;
                        break;
                    }
                }
                break;
            case MSG_ID_MEDIA_PAUSE:
                {
                    StreamMediaPauseReq req;
                    if ( Unpack_MsgStreamMediaPauseReq((char*)msg_buf, msg_len, &req) < 0 )
                    {
                        ret = -2;
                        break;
                    }
                    if ( stream_on_msg_media_pause_req(session, header, &req) < 0 )
                    {
                        ret = -3;
                        break;
                    }
                }
                break;
            case MSG_ID_MEDIA_CMD:
                {
                    StreamMediaCmdReq req;
                    if ( Unpack_MsgStreamMediaCmdReq((char*)msg_buf, msg_len, &req) < 0 )
                    {
                        ret = -2;
                        break;
                    }
                    if ( stream_on_msg_media_cmd_req(session, header, &req) < 0 )
                    {
                        ret = -3;
                        break;
                    }
                }
                break;
            case MSG_ID_MEDIA_CLOSE:
                {
                    StreamMediaCloseReq req;
                    if ( Unpack_MsgStreamMediaCloseReq((char*)msg_buf, msg_len, &req) < 0 )
                    {
                        ret = -2;
                        break;
                    }
                    if ( stream_on_msg_media_close_req(session, header, &req) < 0 )
                    {
                        ret = -3;
                        break;
                    }
                }
                break;
            default:
                {
                    ret = -4;
                }
                break;
            }
        }
        else if(header->msg_type == MSG_TYPE_RESP)
        {
            switch(header->msg_id)
            {
            case MSG_ID_MEDIA_CONNECT:
                {
                    StreamMediaConnectResp resp;
                    if ( Unpack_MsgStreamMediaConnectResp((char*)msg_buf, msg_len, &resp) < 0 )
                    {
                        ret = -2;
                        break;
                    }

                    if ( stream_on_msg_media_connect_resp(session, &resp) < 0 )
                    {
                        ret = -3;
                        break;
                    }
                }
                break;
            case MSG_ID_MEDIA_DISCONNECT:
                {
                    StreamMediaDisconnectResp resp;
                    if ( Unpack_MsgStreamMediaDisconnectResp((char*)msg_buf, msg_len, &resp) < 0 )
                    {
                        ret = -2;
                        break;
                    }

                    if ( stream_on_msg_media_disconnect_resp(session, &resp) < 0 )
                    {
                        ret = -3;
                        break;
                    }
                }
                break;
            case MSG_ID_MEDIA_STATUS:
                {
                    StreamMediaStatusResp resp;
                    if ( Unpack_MsgStreamMediaStatusResp((char*)msg_buf, msg_len, &resp) < 0 )
                    {
                        ret = -2;
                        break;
                    }

                    if ( stream_on_msg_media_status_report_resp(session, &resp) < 0 )
                    {
                        ret = -3;
                        break;
                    }
                }
                break;
            default:
                {
                    ret = -5;
                }
                break;
            }
        }
        else
        {
            ret = -6;
        }

    } while (0);

    return ret;
}

static int stream_session_on_tp_open(void *arg)
{
    int rst = RS_KP;
    vos_uint32_t cur_tim;
	media_session_t* session = (media_session_t*)arg;
	
    if( !session )
    {
        DNL_ERROR_LOG("stream_on_tp_open-->connector is nil!\n");
        return RS_NG;
    }

    switch(session->sub_seq)
    {
        case 0:
            if(stream_session_transport_open(session) < 0)
        	{
        		DNL_ERROR_LOG("stream_on_tp_open-->open failed!\n");
        		rst = RS_NG;
        	}
            else
            {
                if(session->tp.tcp_state == en_tcp_state_connected)
                {
                    rst = RS_OK;
                    DNL_INFO_LOG("stream_on_tp_open-->open success, sock_fd=%d.\n", session->tp.sock);
                }
                else if(session->tp.tcp_state == en_tcp_state_connecting)
                {
                    session->sub_start_time = vos_get_system_tick_sec();
                    session->sub_seq ++;
                    rst = RS_KP;
                    DNL_ERROR_LOG("stream_on_tp_open-->connecting, sock_fd=%d.\n", session->tp.sock);
                }
                else
                {
                    rst = RS_NG;
    		        DNL_ERROR_LOG("stream_on_tp_open-->open failed, tcp_state=%d!\n", session->tp.tcp_state);
                }
            }
            break;
        case 1:
            if(dnl_transport_tcp_connect_check(&session->tp) == 0)
            {
                rst = RS_OK;
                DNL_ERROR_LOG("stream_on_tp_open-->open success, sock_fd=%d.\n", session->tp.sock);
            }
            else
            {
                cur_tim = vos_get_system_tick_sec();
                if( (cur_tim - session->sub_start_time) > 10 )
                {
                    rst = RS_NG;
    		        DNL_ERROR_LOG("stream_on_tp_open-->open timeout, sock_fd=%d.\n", session->tp.sock);
                }
                else
                {
                    rst = RS_KP;
                }
            }
            break;
        default:
            rst = RS_NG;
            DNL_ERROR_LOG("session open failed, subseq=%d!\n", session->sub_seq);
            break;
    }
	
	return rst;
}

static int stream_session_on_exchange(void *arg)
{
	int rst = RS_KP;
	media_session_t* session = (media_session_t*)arg;

    do 
    {
        vos_uint32_t cur_tim;
        tp_tcp_state_e tcp_state;

        tcp_state = dnl_transport_tcp_state( &session->tp );
        if ( tcp_state != en_tcp_state_connected )
        {
            DNL_ERROR_LOG(
                "stream_on_exchange-->01 error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );

            rst = RS_NG;
            break;
        }

        if( !STREAM_ENCRYPT ) //不加密无需密钥交换
        {
            rst = RS_OK;
            break;
        }

        switch(session->sub_seq)
        {
        case 0:
            {
                vos_uint32_t req_seq = ++session->req_seq;
                if(dnl_tp_tx_data(&session->tp, MSG_ID_EXCHANGE_KEY, MSG_TYPE_REQ, &req_seq, 0) == 0)
                {
                    session->sub_seq++;
                    session->sub_start_time = vos_get_system_tick_sec();
                    rst = RS_KP;
                    DNL_ERROR_LOG("stream_on_exchange-->connecting, sock_fd=%d.\n", session->tp.sock);
                }
                else
                {
                    rst = RS_NG;
                    DNL_ERROR_LOG("stream_on_exchange-->exchange failed!\n");
                }
            }
            break;
        case 1:
            {
                int ret = dnl_tp_rx_data(&session->tp, MSG_ID_EXCHANGE_KEY, MSG_TYPE_RESP, session->req_seq);
                if(ret == en_tp_rx_recving)
                {
                    cur_tim = vos_get_system_tick_sec();
                    if( (cur_tim - session->sub_start_time) > 10)
                    {
                        rst = RS_NG;
                        DNL_ERROR_LOG("stream_on_exchange-->exchange timeout, sock_fd=%d.\n", session->tp.sock);
                    }
                }
                else if(ret == en_tp_rx_complete)
                {
                    rst = RS_OK;
                    DNL_ERROR_LOG("stream_on_exchange-->exchange success, sock_fd=%d.\n", session->tp.sock);
                }
                else
                {
                    rst = RS_NG;
                    DNL_ERROR_LOG("stream_on_exchange-->exchange failed!\n");
                }
            }
            break;
        default:
            {
                session->sub_seq = 0;
                DNL_ERROR_LOG("stream_on_exchange-->default, sock_fd=%d, subseq=%d.\n", session->tp.sock, session->sub_seq);
            }            
            break;
        }

    } while (0);
    
    if(rst==RS_OK)
    {
        stream_session_status_change(session, en_stream_session_on_connecting);
    }
    else if(rst==RS_NG)
    {
        stream_session_status_change(session, en_stream_session_on_close);
    }

	return rst;
}

static int stream_session_on_media_connect(void *arg)
{
    int rst = RS_KP;
	vos_uint32_t cur_tim;
	tp_tcp_state_e tcp_state;
	
	media_session_t* session = (media_session_t*)arg;
    do 
    {
        tcp_state = dnl_transport_tcp_state( &session->tp );
        if ( tcp_state != en_tcp_state_connected )
        {
            DNL_ERROR_LOG(
                "error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );
            rst = RS_NG;
            break;
        }

        switch(session->sub_seq)
        {
        case 0:
            {
                if(dnl_tp_tx_data(&session->tp, MSG_ID_MEDIA_CONNECT, MSG_TYPE_REQ, session, 0) == 0)
                {
                    session->sub_seq ++;
                    session->sub_start_time = vos_get_system_tick_sec();
                    rst = RS_KP;
                    DNL_ERROR_LOG("stream_on_login-->connecting, sock_fd=%d.\n", session->tp.sock);
                }
                else
                {
                    rst = RS_NG;
                    DNL_ERROR_LOG("stream_on_login-->login failed!\n");
                }
            }
            break;
        case 1:
            {
                int ret = dnl_tp_rx_data(&session->tp, MSG_ID_MEDIA_CONNECT, MSG_TYPE_RESP, session->req_seq);
                if(ret == en_tp_rx_recving)
                {
                    cur_tim = vos_get_system_tick_sec();
                    if( (cur_tim - session->sub_start_time) > 10)
                    {
                        rst = RS_NG;
                        DNL_ERROR_LOG("stream_on_login-->login timeout, sock_fd=%d.\n", session->tp.sock);
                    }
                }
                else if(ret == en_tp_rx_complete)
                {
                    rst = RS_OK;
                    DNL_ERROR_LOG("stream_on_login-->login success, sock_fd=%d.\n", session->tp.sock);
                }
                else
                {
                    rst = RS_NG;
                    DNL_ERROR_LOG("stream_on_login-->login failed!\n");
                }
            }
            break;
        default:
            {
                session->sub_seq = 0;
                DNL_ERROR_LOG("stream_on_login-->default, sock_fd=%d, subseq=%d.\n", session->tp.sock, session->sub_seq);
            }            
            break;
        }
    } while (0);
    
    if(rst==RS_OK)
    {
        if (session->is_pause)
        {
            stream_session_status_change(session, en_stream_session_on_pause);
        }
        else
        {
            stream_session_status_change(session, en_stream_session_on_play);
        }
    }
    else if(rst==RS_NG)
    {
        stream_session_status_change(session, en_stream_session_on_close);
    }
	
	return rst;
}

static int stream_session_on_normal_run(void *arg)
{
    int rst = RS_KP;
	tp_tcp_state_e tcp_state;
	media_session_t* session = (media_session_t*)arg;

    do 
    {
        //01. check tcp state
        tcp_state = dnl_transport_tcp_state( &session->tp );
        if ( tcp_state != en_tcp_state_connected )
        {
            DNL_ERROR_LOG(
                "error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );
            rst = RS_NG;
            break;
        }

        //02. check tcp remained tx data
        if( dnl_transport_tx_writing(&session->tp) )
        {
            if( dnl_transport_send(&session->tp) < 0 )
            {
                DNL_DEBUG_LOG(
                    "stream_on_normal_run-->send remained data fail, sock_fd=%d, data_size=%d, sent_size=%d.\n", 
                    session->tp.sock, 
                    session->tp.tx.data_size, 
                    session->tp.tx.sent_size );
                rst = RS_NG;
                break;
            }

            break;
        }

        //03. handle rx data
        {
            if( dnl_tp_rx_data(&session->tp, 0, 0, 0 ) == en_tp_rx_err)
            {
                DNL_ERROR_LOG(
                    "receive error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d.\n", 
                    session->tp.sock, 
                    tcp_state, 
                    session->tp.tx.flag, 
                    session->tp.rx.flag );
                rst = RS_NG;
                break;
            }
        }

        //04. status report and media dispactch
        if( dnl_transport_tx_enable(&session->tp) )
        {
            if ( stream_session_status_report(session) != 0)
            {
                DNL_ERROR_LOG("report status error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d.\n", 
                    session->tp.sock, 
                    tcp_state, 
                    session->tp.tx.flag, 
                    session->tp.rx.flag );
                rst = RS_NG;
                break;
            }

            if ( stream_session_frame_report(session) != 0 )
            {
                DNL_ERROR_LOG(
                    "dispatch session media error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d.\n", 
                    session->tp.sock, 
                    tcp_state, 
                    session->tp.tx.flag, 
                    session->tp.rx.flag );
                rst = RS_NG;
                break;
            }
        }
        else
        {
            DNL_ERROR_LOG("dnl_transport_tx_enable failed, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d.\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );
            //rst = RS_NG;
        }

    } while (0);
    
    if(rst==RS_NG)
    {
        stream_session_status_change(session, en_stream_session_on_close);
    }
    else
    {
        session->retry_cnt = 0;
    }

	return rst;
}

static int stream_session_on_close(void *arg)
{
	int ret = RS_OK;

	media_session_t* session = (media_session_t*)arg;

	DNL_DEBUG_LOG("stream_on_close-->sock_fd=%d, retry_cnt=%d, tx_flag=%d, rx_flag=%d.\n", 
		session->tp.sock, session->retry_cnt,session->tp.tx.flag, session->tp.rx.flag);

	Clear_DH_conn_status( session->tp.sock );

	dnl_transport_close(&session->tp);

    if (++session->retry_cnt > 5)
    {
        session->running = FALSE;
        entry_media_session_close_notify(session->session_type, session->channel_id, session->stream_id);
    }

	return ret;
}

static int stream_on_msg_media_connect_resp(media_session_t *session, StreamMediaConnectResp* resp)
{
    int ret = 0;
    do 
    {
        if (resp->resp_code != EN_SUCCESS)
        {
            ret = -1;
            break;
        }

        if (!resp->mask&0x01)
        {
            ret = -2;
            break;
        }

        if ( strcmp(session->session_id, resp->session_id) != 0 )
        {
            ret = -3;
            break;
        }

    } while (0);

    return ret;
}

static int stream_on_msg_media_disconnect_resp(media_session_t *session, StreamMediaDisconnectResp* resp)
{
    int ret = 0;
    do 
    {
        if (resp->resp_code != EN_SUCCESS)
        {
            ret = -1;
            break;
        }

        if (!resp->mask&0x01)
        {
            ret = -2;
            break;
        }

        if ( strcmp(session->session_id, resp->session_id) != 0 )
        {
            ret = -3;
            break;
        }

    } while (0);

    return ret;
}

static int stream_on_msg_media_status_report_resp(media_session_t *session, StreamMediaStatusResp* resp)
{
    int ret = 0;
    do 
    {
        if (resp->resp_code != EN_SUCCESS)
        {
            ret = -1;
            break;
        }

        if (!resp->mask&0x01)
        {
            ret = -2;
            break;
        }

        if ( strcmp(session->session_id, resp->session_id) != 0 )
        {
            ret = -3;
            break;
        }

    } while (0);

    return ret;
}

static int stream_on_msg_media_play_req(media_session_t *session, MsgHeader* header, StreamMediaPlayReq* req)
{
    int ret = 0;

    DNL_TRACE_LOG(
        "stream_on_msg_media_play_req, session_id=%s, mask=%u, status=%d ->2\n", 
        req->session_id, 
        req->mask,
        session->status);

    do 
    {
        StreamMediaPlayResp resp;
        memset(&resp, 0x0, sizeof(resp));

        if (!req->mask&0x01)
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -1;
            break;
        }

        if ( strcmp(session->session_id, req->session_id) != 0 )
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -2;
            break;
        }

        if ( session->status==en_stream_session_on_pause )
        {
            session->is_pause = false;
            stream_session_status_change(session, en_stream_session_on_play);
        }

        resp.resp_code = EN_SUCCESS;
        resp.mask = 0x01;
        strncpy(resp.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);

        //发送响应
        if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
        {
            ret = -3;
            break;
        }

    } while (0);

    return ret;
}

static int stream_on_msg_media_pause_req(media_session_t *session, MsgHeader* header, StreamMediaPauseReq* req)
{
    int ret = 0;

    DNL_TRACE_LOG(
        "stream_on_msg_media_pause_req, session_id=%s, mask=%u, status=%d ->3\n", 
        req->session_id, 
        req->mask,
        session->status);

    do 
    {
        StreamMediaPauseResp resp;
        memset(&resp, 0x0, sizeof(resp));

        if (!req->mask&0x01)
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -1;
            break;
        }

        if ( strcmp(session->session_id, req->session_id) != 0 )
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -2;
            break;
        }

        if ( session->status==en_stream_session_on_play )
        {
            session->is_pause = true;
            stream_session_status_change(session, en_stream_session_on_pause);
        }

        resp.resp_code = EN_SUCCESS;
        resp.mask = 0x01;
        strncpy(resp.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);
        
        //发送响应
        if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
        {
            ret = -3;
            break;
        }

    } while (0);

    return ret;
}

static int stream_on_msg_media_cmd_req(media_session_t *session, MsgHeader* header, StreamMediaCmdReq* req)
{
    int ret = 0;

    DNL_TRACE_LOG(
        "stream_on_msg_media_cmd_req, session_id=%s, mask=%u, cmd_type=%u, status=%d\n", 
        req->session_id, 
        req->mask,
        req->cmd_type,
        session->status);

    do 
    {
        StreamMediaCmdResp resp;
        memset(&resp, 0x0, sizeof(resp));

        if (!req->mask&0x01)
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -1;
            break;
        }

        if ( strcmp(session->session_id, req->session_id) != 0 )
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -2;
            break;
        }

        if ( req->cmd_type==MEDIA_CMD_VIDEO_OPEN )
        {
            session->is_video_open = TRUE;
        }
        else if ( req->cmd_type==MEDIA_CMD_VIDEO_CLOSE )
        {
            session->is_video_open = FALSE;
        }
        else if ( req->cmd_type==MEDIA_CMD_AUDIO_OPEN )
        {
            session->is_audio_open = TRUE;
        }
        else if ( req->cmd_type==MEDIA_CMD_AUDIO_CLOSE )
        {
            session->is_audio_open = FALSE;
        }
        else
        {

        }

        resp.resp_code = EN_SUCCESS;
        resp.mask = 0x01;
        strncpy(resp.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);
        
        //发送响应
        if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
        {
            ret = -3;
            break;
        }

    } while (0);

    return ret;
}

static int stream_on_msg_media_close_req(media_session_t *session, MsgHeader* header, StreamMediaCloseReq* req)
{
    int ret = 0;

    DNL_TRACE_LOG(
        "stream_on_msg_media_close_req, session_id=%s, mask=%u, status=%d ->4\n", 
        req->session_id, 
        req->mask,
        session->status);

    do 
    {
        StreamMediaCloseResp resp;
        memset(&resp, 0x0, sizeof(resp));

        if (!req->mask&0x01)
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -1;
            break;
        }

        if ( strcmp(session->session_id, req->session_id) != 0 )
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            ret = -2;
            break;
        }

        //session->is_pause = true;
        session->running = FALSE;
        stream_session_status_change(session, en_stream_session_on_close);
        entry_media_session_close_notify(session->session_type, session->channel_id, session->stream_id);

        resp.resp_code = EN_SUCCESS;
        resp.mask = 0x01;
        strncpy(resp.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);
        
        //发送响应
        if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
        {
            ret = -3;
            break;
        }

    } while (0);

    return ret;
}

static int stream_session_conttbl_proc(media_session_t* session)
{
    int ret = RS_KP;
    vos_uint32_t s_tick, e_tick, delt_tick;
    vos_int32_t cur_cont_seq = 0;
    //vos_time_val tv = {0, 0};
    dnl_conttbl_t *cont_tbl;
    
    s_tick = vos_get_system_tick();
    cur_cont_seq = session->cont_seq;

    if(!session->running)
    {
        return RS_NG;
    }

    cont_tbl = &stream_session_conttbl_tbl[session->cont_seq];
    ret = cont_tbl->func( session );
    if( ret == RS_OK )
    {
        session->cont_seq = cont_tbl->ok_seq;
        session->sub_seq = 0;
    }
    else if( ret == RS_NG )
    {
        session->cont_seq = cont_tbl->ng_seq;
        session->sub_seq = 0;
    }
    else
    {
    }

    e_tick = vos_get_system_tick();
    delt_tick = e_tick - s_tick;
    if( delt_tick > 200 )
    {
        DNL_WARN_LOG(
            "stream session handle time too long, (%s, %u, %d, %d), (%u, %u, %u)!\n",
            session->session_id, cur_cont_seq, session->status, ret, 
            s_tick, e_tick, delt_tick);
    }

    return ret;
}

static int stream_session_transport_open(media_session_t* session)
{
    addr_info* svr_addr = NULL;
    dnl_transport_cfg_t tp_cfg;

    if(!session)
    {
        return -1;
    }

    memset(&tp_cfg, 0x0, sizeof(tp_cfg));

    svr_addr = &session->stream_svr_list[session->cur_stream_svr_idx];
    tp_cfg.peer_host.sin_addr = svr_addr->sin_addr;
    tp_cfg.peer_host.sin_port = vos_htons( svr_addr->port );
    tp_cfg.sock_type = SOCK_STREAM;
    if( dnl_transport_open(&tp_cfg, &session->tp) != 0)
    {
        return -1;
    }

    if ( session->tp.tx.buf == NULL )
    {
        session->tp.tx.buf = VOS_MALLOC_BLK_T(vos_uint8_t, STREAM_TX_BUF_LEN);
        session->tp.tx.max_size = STREAM_TX_BUF_LEN;
    }

    if ( session->tp.rx.buf == NULL )
    {
        session->tp.rx.buf = VOS_MALLOC_BLK_T(vos_uint8_t, STREAM_RX_BUF_LEN);
        session->tp.rx.max_size = STREAM_RX_BUF_LEN;
    }

    return 0;
}

static void stream_session_status_change(media_session_t* session, stream_session_status_e status)
{
    int last_status = session->status;
    if( ( (last_status == en_stream_session_on_play || last_status == en_stream_session_on_pause) && status == en_stream_session_on_close)
        || (last_status == en_stream_session_on_connecting && (status == en_stream_session_on_play || status == en_stream_session_on_pause) ) )
    {
        DeviceMediaSessionStatus media_status;
        strncpy( media_status.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);
        media_status.session_type = session->session_type;
        strncpy( media_status.device_id, g_DnlDevInfo.dev_id, MAX_DEV_ID_LEN);
        media_status.channel_id = session->channel_id;
        media_status.stream_id = session->stream_id;

        if ( status == en_stream_session_on_close )
        {
            media_status.session_status = 0;
        }
        else if ( status == en_stream_session_on_play || status == en_stream_session_on_pause )
        {
            media_status.session_status = 2;
        }

        {
            addr_info* stream_svr_addr = &session->stream_svr_list[session->cur_stream_svr_idx];
            strncpy( media_status.stream_addr.ip, stream_svr_addr->IP, MAX_IP_LEN);
            media_status.stream_addr.port = stream_svr_addr->port;
        }

        entry_media_staus_chg(media_status);
    }
    session->status = status;
    DNL_TRACE_LOG("stream_session_status_change, status %d -> %d\n", session->status, status );
}

static int stream_session_status_report(media_session_t *session)
{
    int ret = 0;
	vos_time_val cur_tim;
	vos_uint32_t repcycle = 10;

	if ( !dnl_transport_tx_enable(&session->tp) )
    {
        return 0;
    }
    
	if ( session->report_cycle == 0 )
	{
		repcycle = 30;
	}
    else
    {
        repcycle = session->report_cycle;
    }
	
	vos_gettimeofday(&cur_tim); 

	if ( (cur_tim.sec - session->last_report_time) >= repcycle )
	{
		if ( dnl_tp_tx_data( &session->tp, MSG_ID_MEDIA_STATUS, MSG_TYPE_REQ, session, 0 ) != 0 )
		{
		    DNL_ERROR_LOG(
                "stream_status_report error, sock_fd=%d, cur_tim=%d, last_reptim=%d, repcycle=%d\n", 
		        session->tp.sock, 
                cur_tim.sec, 
                session->last_report_time, 
                repcycle );
			return 0;
		}

		DNL_TRACE_LOG(
            "send media status report success, sock_fd=%d, cur_tim=%d, last_reptim=%d, repcycle=%d\n", 
		    session->tp.sock, 
            cur_tim.sec, 
            session->last_report_time, 
            repcycle );

		session->last_report_time = cur_tim.sec;

        //for test
        /*{
            vos_uint32_t cur_tick = vos_get_system_tick();
            if(cur_tick - session->start_time >56*1000)
            {
                ret = -1;
            }
        }*/
	}

	return ret;
}

static int stream_session_frame_report(media_session_t *session)
{
    int ret = 0;

    do 
    {
        //vos_sock_t fd_max = 0;
        fd_set  select_fds;
        struct timeval t_timeout = {0, 0};
        //volatile vos_uint32_t media_que_frm_num = 0;

        if ( session->status != en_stream_session_on_play )
        {
            //DNL_ERROR_LOG("stream_session_frame_report, status=%d", session->status);
            ret = 0;
            break;
        }

        if ( !dnl_transport_tx_enable(&session->tp) )
        {
            DNL_ERROR_LOG("stream_session_frame_report, dnl_transport_tx_enable=0");
            ret = 0;
            break;
        }

        FD_ZERO(&select_fds);
        FD_SET(session->tp.sock, &select_fds);

        ret = select(session->tp.sock+ 1, NULL, &select_fds, NULL, &t_timeout);
        if(ret>0 && FD_ISSET(session->tp.sock, &select_fds))
        {
            media_queue_t *media_que = &session->media_frame.media_que;

            if ( media_que->frm_nums == 0)
            {
                ret = 0;
                break;
            }

            {
                int ret = 0;
                MsgHeader msg_header;
                StreamMediaFrameNotify msg_body;
                {
                    msg_body.mask = 0x01;
                    strncpy(msg_body.session_id, session->session_id, MAX_MEDIA_SESSION_ID_LEN);
                }
                {
                    media_frame_data_t *frame_data = &media_que->que[media_que->rd_idx];
                    msg_body.mask |= 0x02;
                    msg_body.frame_type = frame_data->frm_type;
                    msg_body.frame_av_seq = frame_data->frm_av_id;
                    msg_body.frame_seq = frame_data->frm_id;
                    msg_body.frame_base_time = frame_data->frm_base_time;
                    msg_body.frame_ts = frame_data->frm_ts;
                    msg_body.frame_size = frame_data->frm_size;

                    msg_body.mask |= 0x08;
                    msg_body.offset = frame_data->frm_offset;
                    msg_body.data_size = frame_data->frm_data_size;
                    memcpy(msg_body.datas, frame_data->frm_data, frame_data->frm_data_size);

                    /*
                    DNL_TRACE_LOG(
                        "send media frame, frame_type=0x%x, frame_av_seq=%u, frame_seq=%u, frame_base_time=%u,frame_ts=%u, frame_size=%u, offset=%u, data_size=%u\n", 
                        frame_data->frm_type, 
                        frame_data->frm_av_id, 
                        frame_data->frm_id, 
                        frame_data->frm_base_time,
                        frame_data->frm_ts,
                        frame_data->frm_size,
                        frame_data->frm_offset,
                        frame_data->frm_data_size
                        );*/
                }

                ret = Pack_MsgStreamMediaFrameNotify((char*)session->tp.tx.buf+sizeof(MsgHeader), session->tp.tx.max_size, &msg_body);
                if(ret<=0)
                {
                    ret = -2;
                    break;
                }
                msg_header.msg_size = ret + sizeof(MsgHeader);
                msg_header.msg_id = MSG_ID_MEDIA_FRAME;
                msg_header.msg_type = MSG_TYPE_NOTIFY;
                msg_header.msg_seq = ++session->req_seq;
                Pack_MsgHeader((char*)session->tp.tx.buf, sizeof(MsgHeader), &msg_header);
                session->tp.tx.data_size = msg_header.msg_size;

                if(media_que->rd_idx >= (MEDIA_QUEUE_SIZE-1) )
                {
                    media_que->rd_idx = 0;
                }
                else
                {
                    media_que->rd_idx ++;
                }

                vos_enter_critical_section();
                media_que->frm_nums --;
                vos_leave_critical_section();
            }

            if ( dnl_tp_tx_data(&session->tp, MSG_ID_MEDIA_FRAME, MSG_TYPE_NOTIFY, session, 0) < 0 )
            {
                ret = -3;
                break;
            }
            
            ret = 0;
        }
        else
        {
            int err = vos_get_native_netos_error();
            if( ( err == 0 ) || ( err == VOS_EWOULDBLOCK) || ( err == VOS_EINTR ) ) 
            {
                break;
            }
            ret = -4;
            DNL_ERROR_LOG("stream_session_frame_report, select sock failed, err=%d!", err);
        }
    } while (0);
    
    return ret;
}

static int stream_media_frame_enque(media_session_t* session, media_frame_data_t *frame_pkg)
{
    vos_uint32_t rw_idx;
    media_queue_t *media_que = &session->media_frame.media_que; 

	vos_enter_critical_section();
	if ( media_que->frm_nums >= MEDIA_QUEUE_SIZE )
	{
		#if DEBUG_FLAG
        session->running_flag = false;
        #endif
		vos_leave_critical_section();
		return -1;
	}
	vos_leave_critical_section();

    rw_idx = media_que->rw_idx;
    memcpy(&media_que->que[rw_idx], frame_pkg, sizeof(media_frame_data_t)); 

    if(media_que->rw_idx >= (MEDIA_QUEUE_SIZE-1))
    {
        media_que->rw_idx = 0;
    }
    else
    {
        media_que->rw_idx++;
    }
	vos_enter_critical_section();
    media_que->frm_nums ++;
	vos_leave_critical_section();

	return 0;
}
