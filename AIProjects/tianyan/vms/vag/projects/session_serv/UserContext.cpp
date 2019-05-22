#include "logging_posix.h"
#include "UserContext.h"
#include "ServerLogical.h"
#include "base/include/TokenMgr.h"
#include "MediaSessionMgr.h"

CUserContext::CUserContext()
{
    running_ = true; //running!!

    send_sink_ = NULL;
    last_active_tick_ = get_current_tick();

    open_media_seq_ = 0;
    session_type_ = 0;
    open_media_tick_ = 0;
}

CUserContext::~CUserContext()
{
    Info("from(%s), user_name(%s), dc(%s), UserContext destroy...", 
        hi_remote_.GetNodeString().c_str(), 
        user_name_.c_str(), 
        dc_.GetString().c_str() );
}

bool CUserContext::IsAlive()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    
    return running_;
}

void CUserContext::OnTcpClose(const CHostInfo& hiRemote)
{
    Debug("from(%s), user_name(%s), dc(%s), disconnect!", 
        hi_remote_.GetNodeString().c_str(), 
        user_name_.c_str(), 
        dc_.GetString().c_str());
    
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if ( session_type_ != 0 )
        {
            (void)GetService()->GetMediaSessionMgr()->OnUserClose( dc_, (SessionType)(int)session_type_, hi_remote_ );
        }

        running_ = false;
        send_sink_ = NULL;
    }
}

void CUserContext::Update()
{
    tick_t curr = get_current_tick();

    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    if( running_ )
    {
        return;
    }

    if( curr - last_active_tick_ > 3*20*1000 )
    {
        running_ = false;

        Error("from(%s), user_name(%s), dc(%s), heartbeat timeout!", 
            hi_remote_.GetNodeString().c_str(), 
            user_name_.c_str(), 
            dc_.GetString().c_str() );

        return;
    }

    if ( open_media_tick_ && ( curr - open_media_tick_ ) > 1000 )
    {
        running_ = false;

        Error("from(%s), user_name(%s), dc(%s), wait media open ack timeout!", 
            hi_remote_.GetNodeString().c_str(),
            user_name_.c_str(),
            dc_.GetString().c_str() );

        return;
    }
}

bool CUserContext::SendMessage(const char* data_buff, uint32 data_size)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    if( !send_sink_  )
    {
        Error(
            "from(%s), user_name(%s), dc(%s), send sink is nil!", 
            hi_remote_.GetNodeString().c_str(), 
            user_name_.c_str(), 
            dc_.GetString().c_str()
            );
        return false;
    }

    send_sink_->SendFunc((uint8*)data_buff, data_size);

    return true;
}

void CUserContext::OnMediaOpenAck(int err_code, const SDeviceChannel& dc, SessionType session_type, const string& session_id, const CHostInfo& hi_stream_serv)
{
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if( running_ )
        {
            return;
        }
    }
    

    CuMediaOpenResp resp;

    do 
    {
        if( err_code != 0 )
        {
            Error("open media failed, user_name(%s), dc(%s), err_code(%d)", 
                user_name_.c_str(), 
                dc.GetString().c_str(), err_code );
            
            resp.mask = 0x00;
            resp.resp_code = err_code;
            break;
        }

        CDeviceContext_ptr pDevCtx = GetService()->GetDeviceContext(dc.device_id_);
        if( !pDevCtx )
        {
            resp.mask = 0x00;
            resp.resp_code = EN_ERR_DEVICE_OFFLINE;
            break;
        }

        protocol::DevChannelInfo channel_info;
        if( !pDevCtx->GetChannel( dc.channel_id_, channel_info ) 
            || channel_info.channel_status == protocol::CHANNEL_STS_OFFLINE 
            || channel_info.stream_num < dc.stream_id_ )
        {
            resp.mask = 0x00;
            resp.resp_code = EN_ERR_CHANNEL_OFFLINE;
            break;
        }

        string s_token;
        CServerLogical* pService = CServerLogical::GetLogical();
        if ( pService->IsEnableTokenAuth() )
        {
            int ret = pService->GetTokenMgr()->StreamUserToken_Gen(user_name_, dc.device_id_, dc.channel_id_, 10, pService->GetAccessKey(),s_token );
            if ( !ret )
            {
                Error( "user(%s), dc(%s), session_id(%s), StreamUserToken_Gen failed.",
                    user_name_.c_str(), dc.GetString().c_str(), session_id.c_str() );
                
                resp.mask = 0x00;
                resp.resp_code = EN_ERR_SERVICE_UNAVAILABLE;
                break;
            }
        }

        resp.mask = 0x00;
        resp.resp_code = 0;

        resp.mask |= 0x01;
        {
            resp.session_id = session_id;
            resp.device_id = dc.device_id_;
            resp.channel_id = dc.channel_id_;
            resp.stream_id = dc.stream_id_;
        }
        
        resp.mask |= 0x02;
        {
            resp.video_codec = channel_info.stream_list[dc.stream_id_].video_codec;
        }

        resp.mask |= 0x04;
        {
            resp.audio_codec = channel_info.audio_codec;
        }

        resp.mask |= 0x08;
        {
            resp.stream_route_table_size = 1;

            protocol::HostAddr stream_host;
            hi_stream_serv.GetIP(stream_host.ip);
            stream_host.port = hi_stream_serv.Port;
            resp.stream_route_tables.push_back(stream_host);

            resp.stream_token.token_bin_length = s_token.length();
            memcpy( resp.stream_token.token_bin, s_token.data(), s_token.length() );
        }

    } while (0);
    
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        open_media_tick_ = 0;

        MsgHeader header;
        header.msg_id = MSG_ID_CU_MEDIA_OPEN;
        header.msg_type = MSG_TYPE_RESP;
        header.msg_seq = open_media_seq_;

        uint8 buf[512];
        CDataStream ds(buf,sizeof(buf));
        ds << header;
        ds << resp;

        *((WORD*)ds.getbuffer()) = ds.size();
        this->SendMessage((const char*)ds.getbuffer(), ds.size());
    }
}

bool CUserContext::ON_CuLoginRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuLoginReq& req,  CuLoginResp& resp )
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        hi_remote_ = hiRemote;
        send_sink_ = sink;

        if(req.mask & 0x01)
        {
            user_name_ = req.user_name;
            token_ = req.token;

            CServerLogical* pService = CServerLogical::GetLogical();
            if ( pService->IsEnableTokenAuth() )
            {
                string s_token( (char*)req.token.token_bin, req.token.token_bin_length );
                CTokenMgr_ptr pTokenMgr = pService->GetTokenMgr();

                if( !pTokenMgr->SessionUserToken_Auth(s_token, user_name_) )
                {
                    resp.resp_code = EN_ERR_TOKEN_CHECK_FAIL;
                    Error("from(%s), user_name(%s), token check error!", 
                        hiRemote.GetNodeString().c_str(), req.user_name.c_str());
                    break;
                }
            }
        }

        if(req.mask & 0x02)
        {
            hi_private_ = CHostInfo(req.private_ip, req.private_port);
        }

        resp.mask = 0;
        resp.resp_code = EN_SUCCESS;
        
        resp.mask = 0x01;
        {
            hi_remote_.GetIP( resp.public_ip );
            resp.public_port = hi_remote_.Port;
        }

    } while (0);

    //send client login response
    {
        MsgHeader header;
        header.msg_id = MSG_ID_CU_LOGIN;
        header.msg_type = MSG_TYPE_RESP;
        header.msg_seq = msg_seq;

        uint8 buf[512];
        CDataStream ds(buf,sizeof(buf));
        ds << header;
        ds << resp;

        *((WORD*)ds.getbuffer()) = ds.size();
        this->SendMessage((const char*)ds.getbuffer(), ds.size());
    }

    Debug(
        "send client login response, from(%s), user_name(%s), dc(%s), resp_code:%d",
        hiRemote.GetNodeString().c_str(), 
        user_name_.c_str(),
        dc_.GetString().c_str(),
        resp.resp_code
        );

    return true;
}

bool CUserContext::ON_CuMediaOpenRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaOpenReq& req )
{
    CuMediaOpenResp resp;

    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);

        if ( !(req.mask & 0x01) )
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            break;
        }

        if (req.session_type == MEDIA_SESSION_TYPE_LIVE)
        {
            session_type_ = en_session_type_live;
        }
        else if (req.session_type == MEDIA_SESSION_TYPE_PU_PLAYBACK)
        {
            session_type_ = en_session_type_playback;
        }
        else
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            break;
        }

        dc_.device_id_ = req.device_id;
        dc_.channel_id_ = req.channel_id;
        dc_.stream_id_ = req.stream_id;

        CMediaSessionMgr_ptr pMediaSessionMgr = GetService()->GetMediaSessionMgr();
        MediaSession_ptr pMediaSession = pMediaSessionMgr->GetMediaSession( dc_, (SessionType)session_type_, true );
        if ( !pMediaSession )
        {
            resp.resp_code = EN_ERR_SERVICE_UNAVAILABLE;
            Error( "from(%s), user_name(%s), dc(%s), create media failed!", 
                hi_remote_.GetNodeString().c_str(), 
                user_name_.c_str(), 
                dc_.GetString().c_str() );
            break;
        }

        if ( ! pMediaSessionMgr->OnUserMediaOpenReq( dc_, (SessionType)session_type_, hiRemote, shared_from_this() ) )
        {
            resp.resp_code = EN_ERR_SERVICE_UNAVAILABLE;
            Error( "from(%s), user_name(%s), dc(%s), connect media session failed!", 
                hi_remote_.GetNodeString().c_str(), 
                user_name_.c_str(), 
                dc_.GetString().c_str() );
            break;
        }

        open_media_seq_ = msg_seq;
        open_media_tick_ = get_current_tick();

        return true;
    } while (0);

    // Send response messages in an abnormal scene
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        open_media_tick_ = 0;

        MsgHeader header;
        header.msg_id = MSG_ID_CU_MEDIA_OPEN;
        header.msg_type = MSG_TYPE_RESP;
        header.msg_seq = msg_seq;

        uint8 buf[512];
        CDataStream ds(buf,sizeof(buf));
        ds << header;
        ds << resp;

        *((WORD*)ds.getbuffer()) = ds.size();
        this->SendMessage((const char*)ds.getbuffer(), ds.size());
    }

    return true;
}

bool CUserContext::ON_CuMediaCloseRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaCloseReq& req,  CuMediaCloseResp& resp )
{
    SDeviceChannel dc;
    dc.device_id_ = req.device_id;
    dc.channel_id_ = req.channel_id;
    dc.stream_id_ = req.stream_id;
    
    CMediaSessionMgr_ptr pMediaSessionMgr = GetService()->GetMediaSessionMgr();
    pMediaSessionMgr->OnUserClose(req.session_id, hiRemote);
    
    MsgHeader header;
    header.msg_id = MSG_ID_CU_MEDIA_CLOSE;
    header.msg_type = MSG_TYPE_RESP;
    header.msg_seq = msg_seq;

    resp.mask = 0;
    resp.resp_code = EN_SUCCESS;

    uint8 buf[512];
    CDataStream ds(buf,sizeof(buf));
    ds << header;
    ds << resp;

    *((WORD*)ds.getbuffer()) = ds.size();
    SendMessage((const char*)ds.getbuffer(), ds.size());

    return true;
}

bool CUserContext::ON_CuStatusReport(ITCPSessionSendSink* sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuStatusReportReq& req, CuStatusReportResp& resp)
{
    Debug(" from(%s), recv cu status report, mask(0x%x).", hiRemote.GetNodeString().c_str(), req.mask);

    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        last_active_tick_ = get_current_tick();

        if (req.mask & 0x01)
        {
            int media_session_num = 0;
            vector<CuMediaSessionStatus> media_sessions;

            media_session_num = req.media_session_num;
            media_sessions = req.media_sessions;
        }

    } while (false);

    resp.mask = 0;
    resp.resp_code = EN_SUCCESS;
    resp.expected_cycle = 30;
    resp.mask |= 0x01;

    MsgHeader header;
    header.msg_id = MSG_ID_CU_STATUS_REPORT;
    header.msg_type = MSG_TYPE_RESP;
    header.msg_seq = msg_seq;

    uint8 msg_buf[256];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();

    SendMessage((const char*)sendds.getbuffer(), sendds.size());

    return true;
}
