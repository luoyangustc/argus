#include "CuStream.h"
#include "ServerLogical.h"

CCuStream::CCuStream()
    : running_( false )
    , send_sink_( NULL )
{
    status_ = en_cu_stream_init;
    send_seq_ = 0;
    video_direct_ = protocol::MEDIA_DIR_RECV_ONLY;
    audio_direct_ = protocol::MEDIA_DIR_SEND_RECV;
    last_active_tick_ = get_current_tick();
}

CCuStream::~CCuStream()
{
}

void CCuStream::Update()
{
    tick_t now = get_current_tick();
    if ( now - last_active_tick_ >= 3*30*1000 )
    {
        status_ = en_cu_stream_error;

        Error( "session_id(%s), session_type(%d), dc(%s), user_name(%s), timeout!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str(),
            user_name_.c_str() );

        return;
    }
}

bool CCuStream::IsAlive()
{
    if ( status_ == en_cu_stream_init ||
        status_ == en_cu_stream_connected ||
        status_ == en_cu_stream_play ||
        status_ == en_cu_stream_pause )
    {
        return true;
    }

    return false;
}

bool CCuStream::IsConnected()
{
    if ( status_ == en_cu_stream_connected ||
        status_ == en_cu_stream_play ||
        status_ == en_cu_stream_pause )
    {
        return true;
    }

    return false;
}

bool CCuStream::Close()
{
    if (!IsConnected())
    {
        return true;
    }

    do
    {
        status_ = en_cu_stream_close;

        MsgHeader header;
        {
            header.msg_id = MSG_ID_MEDIA_CLOSE;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++send_seq_;
        }

        StreamMediaCloseReq req;
        {
            req.mask = 0x01;
            req.session_id = session_id_;
        }

        char msg_buf[512];
        CDataStream sendds(msg_buf, sizeof(msg_buf));
        {
            sendds << header;
            sendds << req;
            *((WORD*)sendds.getbuffer()) = sendds.size();
        }

        if( !SendMsg((const char*)(uint8*)sendds.getbuffer(), sendds.size()) )
        {
            Warn( "session_id(%s), session_type(%d), dc(%s), send media close msg to user_name(%s,%s) failed!", 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str(),
                user_name_.c_str(),
                hi_remote_.GetNodeString().c_str() );
            break;
        }
        Debug( "session_id(%s), session_type(%d), dc(%s), send media close msg to user_name(%s,%s)!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str(),
            user_name_.c_str(),
            hi_remote_.GetNodeString().c_str() );
        return true;
    } while (0);
    return false;
}

bool CCuStream::OnTcpClose()
{
    status_ = en_cu_stream_close;

    Debug( "session_id(%s), session_type(%d), user_name(%s), dc(%s), tcp closed!", 
        session_id_.c_str(),
        session_type_,
        user_name_.c_str(),
        dc_.GetString().c_str() );

    return true;
}

bool CCuStream::OnStream(MI_FrameData_ptr frame_data)
{
    do 
    {
        if(!frame_data)
        {
            Error("from(%s), session_id(%s), user_name(%s), dc(%s), frame data is nil!", 
                hi_remote_.GetNodeString().c_str(), 
                session_id_.c_str(),
                user_name_.c_str(), 
                dc_.GetString().c_str());
            break;
        }

        /*Trace("from(%s), session_id(%s), user_name(%s), dc(%s), audio_open(%d), video_open(%d), status(%d)!", 
            hi_remote_.GetNodeString().c_str(), 
            session_id_.c_str(),
            user_name_.c_str(), 
            dc_.GetString().c_str(),
            audio_open_,
            video_open_,
            status_);*/

        protocol::EnFrameType frame_type = (protocol::EnFrameType)(frame_data->frame_type_&MediaFrameMask);
        if( (frame_type== protocol::FRAME_TYPE_AU) && !audio_open_ )
        {
            break;
        }
        if( (frame_type== protocol::FRAME_TYPE_I || frame_type== protocol::FRAME_TYPE_P) && !video_open_ )
        {
            break;
        }

        if( status_ == en_cu_stream_connected)
        {
            status_ = en_cu_stream_play;
        }
        if (status_ != en_cu_stream_play)
        {
            break;
        }

        if ( session_type_ == MEDIA_SESSION_TYPE_LIVE || 
             session_type_ == MEDIA_SESSION_TYPE_DIRECT_LIVE )
        {
            int sendQLeng = send_sink_->GetSendQLengthFunc();
            if (sendQLeng > 100 && !frame_data->is_i_frame_)
            {
                //break;
            }
        }

        int piece_cnt = (frame_data->frame_size_ + MAX_MSG_BUFF_SIZE - 1) / MAX_MSG_BUFF_SIZE;
        for (int i = 0; i < piece_cnt; i++)
        {
            MsgHeader header;
            {
                header.msg_id = MSG_ID_MEDIA_FRAME;
                header.msg_type = MSG_TYPE_NOTIFY;
                header.msg_seq = ++send_seq_;
            }

            StreamMediaFrameNotify notify;
            {
                notify.mask = 0x01;
                notify.session_id = session_id_;

                notify.mask |= 0x02;
                notify.frame_type = frame_data->frame_type_;
                notify.frame_av_seq = frame_data->frame_av_seq_;
                notify.frame_seq = frame_data->frame_seq_;
                notify.frame_base_time = frame_data->frame_base_time_;
                notify.frame_ts = frame_data->frame_ts_;
                notify.frame_size = frame_data->frame_size_;

                //notify.mask |= 0x04;
                //notify.crc32_hash = frame_data->crc32_hash_;


                uint32 offset = i * MAX_MSG_BUFF_SIZE;
                uint32 length = MAX_MSG_BUFF_SIZE;

                if (offset + length > notify.frame_size)
                {
                    length = notify.frame_size - offset;
                }

                notify.mask |= 0x08;
                notify.offset = offset;
                notify.data_size = length;
                for(uint32 i=0; i<length; ++i)
                {
                    uint8 data = *(frame_data->data_.get()+offset+i);
                    notify.datas.push_back(data);
                }
            }
            
            //uint32 buf_size = sizeof(notify) + sizeof(header);
            //boost::shared_array<uint8> send_buff(new uint8[buf_size]);
            uint8 szSendBuff[2*MAX_MSG_BUFF_SIZE];
            CDataStream sendds(szSendBuff, sizeof(szSendBuff));
            {
                sendds << header;
                sendds << notify;
                *((WORD*)sendds.getbuffer()) = sendds.size();
            }
            
            if( !SendMsg((const char*)(uint8*)sendds.getbuffer(), sendds.size()) )
            {
                Error("peer_host(%s), session_id(%s), user_name(%s), dc(%s), "
                    "frame_type(0x%x),frame_av_seq(%u),frame_seq(%u),frame_base_time(%u),frame_ts(%u),frame_size(%u),offset(%u),data_size(%u)!",
                    "send frame msg failed!", 
                    hi_remote_.GetNodeString().c_str(), 
                    session_id_.c_str(),
                    user_name_.c_str(), 
                    dc_.GetString().c_str(),
                    notify.frame_type, notify.frame_av_seq, notify.frame_seq, notify.frame_base_time, notify.frame_ts, notify.frame_size, notify.offset, notify.data_size);
                break;
            }

            Trace(
                "send frame to cu stream(%s,%s), session_id(%s), session_type(%d), dc(%s), "
                "frame_type(0x%x),frame_av_seq(%u),frame_seq(%u),frame_base_time(%u),frame_ts(%u),frame_size(%u),offset(%u),data_size(%u)!", 
                hi_remote_.GetNodeString().c_str(),
                user_name_.c_str(),
                notify.session_id.c_str(), 
                session_type_, 
                dc_.GetString().c_str(),
                notify.frame_type, 
                notify.frame_av_seq, 
                notify.frame_seq, 
                notify.frame_base_time, 
                notify.frame_ts, 
                notify.frame_size, 
                notify.offset, 
                notify.data_size );
        }

    } while (false);

    return true;
}

bool CCuStream::OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp)
{
    Debug("from(%s), user_name(%s), endpoint_type(%u), session_id(%s), session_type(%d), dc(%s-%u-%u), cu stream connect!", 
        hiRemote.GetNodeString().c_str(),
        req.endpoint_name.c_str(), 
        req.endpoint_type,
        req.session_id.c_str(),
        req.session_type,
        req.device_id.c_str(),
        req.channel_id,
        req.stream_id  );

    do 
    {
        string user_name = req.endpoint_name;
        CDeviceID device_id(req.device_id.c_str());
        c3_crypt_key::STokenSource p_ts;
		CTokenMgr_ptr pTokenMgr = CServerLogical::GetLogical()->GetTokenMgr();
		//TODO: liuzy @Date:2017/5/16 10:05:07 暂时取消验证
        /*if( !pTokenMgr->CheckStreamTokenForClient(req.token, hiRemote, user_name, device_id, req.channel_idx, p_ts) )
        {
            resp.resp_code = EN_ERR_TOKEN_CHECK_FAIL;
            Error("from(%s), user_name(%s), session_id(%s), session_type(%d), dc(%s-%u-%u),token check error!", 
                hiRemote.GetNodeString().c_str(),
                req.endpoint_name.c_str(), 
                req.session_id.c_str(),
                req.session_type,
                req.device_id.c_str(),
                req.channel_idx,
                req.rate_type );
            break;
        }*/

        send_sink_ = sink;
        hi_remote_ = hiRemote;
        user_name_ = req.endpoint_name;
        endpoint_type_ = (EndPointType)req.endpoint_type;

        token_ = req.token;

        session_id_ = req.session_id;
        session_type_ = (EnMediaSessionType)req.session_type;
        video_open_ = (req.session_media & 0x01 || req.session_media & 0x03) ? true : false;
        audio_open_ = (req.session_media & 0x02 || req.session_media & 0x03) ? true : false;

        SDeviceChannel session_dc(req.device_id, req.channel_id, req.stream_id);
        dc_ = session_dc;

        if (req.mask & 0x02)
        {
            video_info_ = req.video_codec;
            video_direct_ = req.video_direct;
        }

        if (req.mask & 0x04)
        {
            audio_info_ = req.audio_codec;
            audio_direct_ = req.audio_direct;
        }

        if (req.mask & 0x08)
        {
            begin_time_ = req.begin_time;
            end_time_ = req.end_time;
        }

        status_ = en_cu_stream_connected;

        last_active_tick_ = get_current_tick();

        return true;
    } while (false);
    return false;
}

bool CCuStream::OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp)
{
    Debug("from(%s), user_name(%s), session_id(%s), session_type(%d), dc(%s), cu stream disconnected!", 
        hiRemote.GetNodeString().c_str(),
        user_name_.c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str() );

    status_ = en_cu_stream_disconnect;

    return true;
}

bool CCuStream::OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp)
{
    Debug("from(%s), user_name(%s),session_id(%s), session_type(%d), dc(%s), cu stream status report!", 
        hiRemote.GetNodeString().c_str(),
        user_name_.c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str() );

    do 
    {
        if ( !req.mask & 0x01)
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            break;
        }

        if ( req.session_id.compare(session_id_) != 0 )
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            break;
        }

        /*
        if (req.mask & 0x02)
        {
            video_open_ = ( 1 == req.video_status ) ? true : false;
        }

        if (req.mask & 0x04)
        {
            audio_open_ = ( 1 == req.audio_status ) ? true : false;
        }*/

        last_active_tick_ = get_current_tick();
        return true;
    } while (0);
    return false;
}

bool CCuStream::OnPlayReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayReq& req, StreamMediaPlayResp& resp)
{
    Debug("from(%s), user_name(%s),session_id(%s), session_type(%d), dc(%s), cu stream play request!", 
        hiRemote.GetNodeString().c_str(),
        user_name_.c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str() );

    status_ = en_cu_stream_play;
    last_active_tick_ = get_current_tick();

    return true;
}

bool CCuStream::OnPauseReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseReq& req, StreamMediaPauseResp& resp)
{
    Debug("from(%s), user_name(%s),session_id(%s), session_type(%d), dc(%s), cu stream pause request!", 
        hiRemote.GetNodeString().c_str(),
        user_name_.c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str() );

    status_ = en_cu_stream_pause;

    last_active_tick_ = get_current_tick();

    return true;
}

bool CCuStream::OnCmdReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdReq& req, StreamMediaCmdResp& resp)
{
    Debug("from(%s), user_name(%s),session_id(%s), session_type(%d), dc(%s), cmd_type(%u), cu stream ctrl request!", 
        hiRemote.GetNodeString().c_str(),
        user_name_.c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str(),
        req.cmd_type );

    switch(req.cmd_type)
    {
    case MEDIA_CMD_VIDEO_OPEN:
        {
            video_open_ = true;
        }
        break;
    case MEDIA_CMD_VIDEO_CLOSE:
        {
            video_open_ = false;
        }
        break;
    case MEDIA_CMD_AUDIO_OPEN:
        {
            audio_open_ = true;
        }
        break;
    case MEDIA_CMD_AUDIO_CLOSE:
        {
            audio_open_ = false;
        }
        break;
    default:
        {
            Warn("from(%s), session_id(%s), session_type(%d), dc(%s), cmd_type(%u), not support!", 
                hiRemote.GetNodeString().c_str(), 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str(),
                req.cmd_type );
            return true;
        }
        break;
    }

    return true;
}

bool CCuStream::SendMsg(const char* msg, uint32 length)
{
    do 
    {
        if (!send_sink_)
        {
            break;
        }

        if( send_sink_->SendFunc((uint8*)msg, length) < 0 )
        {
            break;
        }

        return true;
    } while (0);
    return false;
}