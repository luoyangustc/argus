#include "PuStream.h"
#include "ServerLogical.h"

const int TICK_INTERVAL_ACTIVE_LAST = 3*30*1000;
CPuStream::CPuStream()
    : running_( false )
    , send_sink_( NULL )
{
    status_ = en_pu_stream_init;
    send_seq_ = 0;

    video_open_ = false;
    audio_open_ = false;
    has_audio_info_ = false;

    last_active_tick_ = get_current_tick();
    last_recv_tick_ = 0;
    last_audio_tick_ = 0;
    last_video_tick_ = 0;
}

CPuStream::~CPuStream()
{
}

bool CPuStream::IsAlive()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    bool is_alive = true;
    switch(status_)
    {
    case en_pu_stream_init:
    case en_pu_stream_connected:
    case en_pu_stream_play:
    case en_pu_stream_pause:
        {
            tick_t now = get_current_tick();
            if( now - last_active_tick_ > TICK_INTERVAL_ACTIVE_LAST)
            {
                is_alive = false;
                Error( "session_id(%s), session_type(%d), dc(%s), timeout(%u, %u)!", 
                    session_id_.c_str(),
                    session_type_,
                    dc_.GetString().c_str(),
                    now, (uint32)last_active_tick_);
            }
        }
        break;
    case en_pu_stream_error:
    case en_pu_stream_disconnect:
    case en_pu_stream_closing:
        {
            is_alive = false;
            Error( "session_id(%s), session_type(%d), dc(%s), status error, %d!", 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str(),
                status_ );
        }
        break;
    default:
        {
            is_alive = false;
        }
        break;
    }

    return is_alive;
}

bool CPuStream::IsConnected()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( status_ == en_pu_stream_connected ||
        status_ == en_pu_stream_play ||
        status_ == en_pu_stream_pause )
    {
        return true;
    }

    return false;
}

EnPuStreamStatus CPuStream::GetStatus()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return (EnPuStreamStatus)status_;
}

int CPuStream::GetAudioInfo(protocol::AudioCodecInfo& audio_info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if( !has_audio_info_ )
    {
        return -1;
    }
    audio_info = audio_info_;
    return 0;
}

int CPuStream::GetVideoInfo(protocol::VideoCodecInfo& video_info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    video_info = video_info_;
    return 0;
}

void CPuStream::Update()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    tick_t now = get_current_tick();
    if ( now - last_active_tick_ >= TICK_INTERVAL_ACTIVE_LAST )
    {
        status_ = en_pu_stream_error;
        Error( "session_id(%s), session_type(%d), dc(%s), timeout(%u, %u)!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str(),
            now, last_active_tick_);
        return;
    }

    if ( !IsConnected() )
    {
        return;
    }
    
    if ( status_ == en_pu_stream_play )
    {
        if ( audio_open_ )
        {
            if ( now - last_audio_tick_ >= 10*1000 )
            {
                audio_open_ = false;
            }
        }
        
        if ( video_open_ )
        {
            if ( now - last_video_tick_ >= 10*1000 )
            {
                video_open_ = false;
            }
        }
    }
}

bool CPuStream::Play()
{
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);

        if (!IsConnected())
        {
            break;
        }

        if( status_ == en_pu_stream_play )
        {
            break;
        }

        MsgHeader header;
        {
            header.msg_id = MSG_ID_MEDIA_PLAY;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++send_seq_;
        }        

        StreamMediaPlayReq req;
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
        
        if( !SendMsg((const char*)(uint8*)(sendds.getbuffer()),sendds.size()) )
        {
            Error( "session_id(%s), session_type(%d), dc(%s), send msg failed!", 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str());
            break;
        }

        status_ = en_pu_stream_play;

        if ( audio_open_ )
        {
            last_audio_tick_ = get_current_tick();
        }

        if ( video_open_ )
        {
            last_video_tick_ = get_current_tick();
        }

        Debug( "session_id(%s), session_type(%d), dc(%s), send play request success!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str() );

        return true;
    } while (0);    
    return false;
}

bool CPuStream::Pause()
{
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!IsConnected())
        {
            break;
        }

        MsgHeader header;
        {
            header.msg_id = MSG_ID_MEDIA_PAUSE;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++send_seq_;
        }        

        StreamMediaPauseReq req;
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

        if ( !SendMsg((const char*)(uint8*)(sendds.getbuffer()),sendds.size()) )
        {
            Error( "session_id(%s), session_type(%d), dc(%s), send msg failed!", 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str());
            break;
        }

        status_ = en_pu_stream_pause;

        Debug( "session_id(%s), session_type(%d), dc(%s), send pause request success!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str() );

        return true;
    } while (0);    
    return false;
}

bool CPuStream::AudioCtrl(bool onoff)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!IsConnected())
        {
            break;
        }

        if( audio_open_ == onoff )
        {
            break;
        }
        audio_open_ = onoff;
        if( audio_open_ )
        {
            last_audio_tick_ = get_current_tick();
        }

        MsgHeader header;
        {
            header.msg_id = MSG_ID_MEDIA_CMD;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++send_seq_;
        }        

        StreamMediaCmdReq req;
		{
            req.mask = 0x01;
            req.session_id = session_id_;
            req.cmd_type = onoff ? MEDIA_CMD_AUDIO_OPEN : MEDIA_CMD_AUDIO_CLOSE;
        }
        

        char msg_buf[512];
        CDataStream sendds(msg_buf, sizeof(msg_buf));
        {
            sendds << header;
            sendds << req;
            *((WORD*)sendds.getbuffer()) = sendds.size();
        }
        
        if ( !SendMsg((const char*)(uint8*)(sendds.getbuffer()),sendds.size()) )
        {
            Error( "session_id(%s), session_type(%d), dc(%s), onoff(%d), send msg failed!", 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str(),
                onoff );
            break;
        }

        Debug( "session_id(%s), session_type(%d), dc(%s), onoff(%d), send audio control request success!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str(),
            onoff );


        return true;
    } while (0);
    return false;
}

bool CPuStream::VideoCtrl(bool onoff)
{
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!IsConnected())
        {
            break;
        }

        if( video_open_ == onoff )
        {
            break;
        }
        video_open_ = onoff;
        if( video_open_ )
        {
            last_video_tick_ = get_current_tick();
        }

        MsgHeader header;
        {
            header.msg_id = MSG_ID_MEDIA_CMD;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++send_seq_;
        }

        StreamMediaCmdReq req;
        {
            req.mask = 0x01;
            req.session_id = session_id_;
            req.cmd_type = onoff ? MEDIA_CMD_VIDEO_OPEN : MEDIA_CMD_VIDEO_CLOSE;
        }

        char msg_buf[512];
        CDataStream sendds(msg_buf, sizeof(msg_buf));
        {
            sendds << header;
            sendds << req;
            *((WORD*)sendds.getbuffer()) = sendds.size();
        }

        if( !SendMsg((const char*)(uint8*)(sendds.getbuffer()),sendds.size()) )
        {
            Error( "session_id(%s), session_type(%d), dc(%s), onoff(%d), send msg failed!", 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str(),
                onoff );
            break;
        }

        Debug( "session_id(%s), session_type(%d), dc(%s), onoff(%d), send video control request success!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str(),
            onoff );

        return true;
    } while (0);    
    return false;
}

bool CPuStream::Close()
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!IsConnected())
        {
            break;
        }

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

        if( !SendMsg((const char*)(uint8*)(sendds.getbuffer()),sendds.size()) )
        {
            Error( "session_id(%s), session_type(%d), dc(%s), video_open(%d), send msg failed!", 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str(),
                (int)video_open_ );
            break;
        }

        status_ = en_pu_stream_closing;

        Debug( "session_id(%s), session_type(%d), dc(%s), video_open(%d), send meida close request success!", 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str(),
            (int)video_open_ );

        return true;
    } while (0);
    return false;
}

bool CPuStream::OnTcpClose()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    status_ = en_pu_stream_disconnect;
    send_sink_ = NULL;
    Debug( "session_id(%s), session_type(%d), dc(%s), tcp closed!", 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str() );

    return true;
}

bool CPuStream::GetFrameData(uint32 frm_seq, uint8 frm_type, MI_FrameData_ptr& frmData)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (!frame_mgr_ptr_)
    {
        return false;
    }

    return frame_mgr_ptr_->GetFrameData(frm_seq, frm_type, frmData);
}

bool CPuStream::GetRecentData(stack<MI_FrameData_ptr>& frm_datas)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (!frame_mgr_ptr_)
    {
        return false;
    }

    if (status_ != en_pu_stream_play)
    {
        return false;
    }

    return frame_mgr_ptr_->GetRecentData(frm_datas);
}

bool CPuStream::SendMsg(const char* msg, uint32 length)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!IsConnected())
        {
            break;
        }

        if (!send_sink_)
        {
            break;
        }

        if ( send_sink_->SendFunc((uint8*)msg, length) < 0 )
        {
            break;
        }

        return true;
    } while (0);
    return false;
}

bool CPuStream::OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp)
{
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        c3_crypt_key::STokenSource p_ts;
        CTokenMgr_ptr pTokenMgr = CServerLogical::GetLogical()->GetTokenMgr();
        if( CServerLogical::GetLogical()->GetServCfg()->IsTokenCheck() )
        {
            CTokenMgr_ptr pTokenMgr = CServerLogical::GetLogical()->GetTokenMgr();
            std::string s_token((char*)req.token.token_bin, req.token.token_bin_length);
            if(!pTokenMgr->StreamDeviceToken_Auth(s_token, req.device_id, req.channel_id))
            {
                resp.resp_code = EN_ERR_TOKEN_CHECK_FAIL;
                Error("from(%s), session_id(%s), session_type(%d), dc(%s-%u-%u),token check error!", 
                    hiRemote.GetNodeString().c_str(), 
                    req.session_id.c_str(),
                    req.session_type,
                    req.device_id.c_str(),
                    req.channel_id,
                    req.stream_id );
                break;
            }
        }

        send_sink_ = sink;
        hi_remote_ = hiRemote;
        endpoint_name_ = req.endpoint_name;

        token_ = req.token;

        session_id_ = req.session_id;
        session_type_ = (EnMediaSessionType)req.session_type;
        session_media_ = req.session_media;
        video_open_ = (req.session_media & 0x01 || req.session_media & 0x03) ? true : false;
        audio_open_ = (req.session_media & 0x02 || req.session_media & 0x03) ? true : false;

        SDeviceChannel session_dc( req.device_id, req.channel_id, req.stream_id );
        dc_ = session_dc;

        if (req.mask & 0x02)
        {
            video_info_ = req.video_codec;
            video_direct_ = req.video_direct;
        }

        if (req.mask & 0x04)
        {
            audio_direct_ = req.audio_direct;

            has_audio_info_ = true;
            audio_info_ = req.audio_codec;
        }

        if (req.mask & 0x08)
        {
            begin_time_ = req.begin_time;
            end_time_ = req.end_time;
        }

        frame_mgr_ptr_.reset(new CFrameMgr(dc_));

        status_ = en_pu_stream_connected;

        last_active_tick_ = get_current_tick();

        Debug("from(%s), session_id(%s), session_type(%d), dc(%s), pu stream connected!", 
            hiRemote.GetNodeString().c_str(), 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str() );
        return true;
    } while (false);
    return false;
}

bool CPuStream::OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    status_ = en_pu_stream_disconnect;

    Debug("from(%s), session_id(%s), session_type(%d), dc(%s), pu stream disconnected!", 
        hiRemote.GetNodeString().c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str() );

    return true;
}

bool CPuStream::OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
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

        if ( req.mask & 0x02 )
        {
            video_open_ = (req.video_status==1)?true:false;
        }

        if ( req.mask & 0x04 )
        {
            audio_open_ = (req.audio_status==1)?true:false;
        }

        last_active_tick_ = get_current_tick();

        Debug("from(%s), session_id(%s), session_type(%d), dc(%s), (0x%02x, a:%u, v:%u), pu stream status report!", 
            hiRemote.GetNodeString().c_str(), 
            session_id_.c_str(),
            session_type_,
            dc_.GetString().c_str(),
            req.mask, audio_open_, video_open_);

        return true;
    } while (0);
    return false;
}

bool CPuStream::OnPlayResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (EN_SUCCESS != resp.resp_code)
    {
        status_ = en_pu_stream_error;
    }

    last_active_tick_ = get_current_tick();

    Debug("from(%s), session_id(%s), session_type(%d), dc(%s), pu stream play resp, resp_code(0x%x)!", 
        hiRemote.GetNodeString().c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str(),
        resp.resp_code );

    return true;
}

bool CPuStream::OnPauseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (EN_SUCCESS != resp.resp_code)
    {
        status_ = en_pu_stream_error;
    }

    last_active_tick_ = get_current_tick();

    Debug("from(%s), session_id(%s), session_type(%d), dc(%s), pu stream pause resp, resp_code(0x%x)!", 
        hiRemote.GetNodeString().c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str(),
        resp.resp_code );

    return true;
}

bool CPuStream::OnCmdResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (EN_SUCCESS != resp.resp_code)
    {
        status_ = en_pu_stream_error;
    }

    last_active_tick_ = get_current_tick();

    Debug("from(%s), session_id(%s), session_type(%d), dc(%s), pu stream ctrl resp, resp_code(0x%x)!", 
        hiRemote.GetNodeString().c_str(), 
        session_id_.c_str(),
        session_type_,
        dc_.GetString().c_str(),
        resp.resp_code );

    return true;
}

bool CPuStream::OnFrameNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaFrameNotify& notify)
{
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!frame_mgr_ptr_)
        {
            Error("from(%s), session_id(%s), session_type(%d), dc(%s), frame mgr is nil!", 
                hiRemote.GetNodeString().c_str(), 
                session_id_.c_str(),
                session_type_,
                dc_.GetString().c_str() );
            break;
        }

        if (!frame_mgr_ptr_->OnStream(notify))
        {
            Error(
                "from(%s),session_id(%s),session_type(%d),dc(%s),frame_type(0x%x),frame_av_seq(%u),frame_seq(%u),frame_base_time(%u),frame_ts(%u), frame_size(%u), offset(%u), data_size(%u), , frame mgr handle stream failed!!", 
                hiRemote.GetNodeString().c_str(), notify.session_id.c_str(), session_type_, dc_.GetString().c_str(),
                notify.frame_type, notify.frame_av_seq, notify.frame_seq, notify.frame_base_time, notify.frame_ts, notify.frame_size, notify.offset, notify.data_size );

            break;
        }

        Trace(
            "from(%s), session_id(%s), session_type(%d), dc(%s), frame_type(0x%x), frame_av_seq(%u), frame_seq(%u),frame_base_time(%u),frame_ts(%u), frame_size(%u), offset(%u), data_size(%u)!", 
            hiRemote.GetNodeString().c_str(), notify.session_id.c_str(), session_type_, dc_.GetString().c_str(),
            notify.frame_type, notify.frame_av_seq, notify.frame_seq, notify.frame_base_time,notify.frame_ts, notify.frame_size, notify.offset, notify.data_size );

        tick_t now = get_current_tick();

        protocol::EnFrameType frame_type = (protocol::EnFrameType)( notify.frame_type & MediaFrameMask );
        if( frame_type== protocol::FRAME_TYPE_AU )
        {
            last_audio_tick_ = now;
        }
        else if( frame_type== protocol::FRAME_TYPE_I || frame_type== protocol::FRAME_TYPE_P )
        {
            last_video_tick_ = now;
        }
        else
        {
            break;
        }

        last_active_tick_ = now;

        return true;
    } while (0);    
    return false;
}

void CPuStream::DumpInfo(Variant& info)
{
    int len = 0;
    char tmpBuf[128]={0};
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    info["remote_host"] = hi_remote_.GetNodeString();
    info["endpoit_name"] = endpoint_name_;

    len = sprintf(tmpBuf, "%d(0:init 1:connected 2:play 3:pause 4:closing 5:disconnect 6:error )", status_);
    if(len < 0)
    {
        info["status"] = status_;
    }
    else
    {
        tmpBuf[len] = '\0';
        info["status"] = tmpBuf;
    }

    len = sprintf(tmpBuf, "open_flag:%d, fmt:%d(0:H264 1:H265)", video_open_, video_info_.codec_fmt);
    if(len < 0)
    {
        info["video_open"] = video_open_;
        info["video_fmt"] = video_info_.codec_fmt;
    }
    else
    {
        tmpBuf[len] = '\0';
        info["video_info"] = tmpBuf;
    }

    len = sprintf(tmpBuf, "open_flag:%d,fmt:%d,channel:%d,sample:%d,bitwidth:%d", 
                audio_open_, audio_info_.codec_fmt, audio_info_.channel, audio_info_.sample, audio_info_.bitwidth);
    if(len < 0)
    {
        info["audio_open"] = audio_open_;
        info["audio_fmt"] = audio_info_.codec_fmt;
    }
    else
    {
        tmpBuf[len] = '\0';
        info["audio_info"] = tmpBuf;
    }

    info["last_active_tick"] = last_active_tick_;
    info["last_audio_tick"] = last_audio_tick_;
    info["last_video_tick"] = last_video_tick_;
    
    if(frame_mgr_ptr_)
    {
        Variant frm_mgr_info;
        frame_mgr_ptr_->DumpInfo(frm_mgr_info);
        info["frame_mgr_info"] = frm_mgr_info;
    }
}
