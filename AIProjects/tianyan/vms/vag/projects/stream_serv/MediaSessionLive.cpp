#include "MediaSessionLive.h"
#include "ServerLogical.h"
#include "to_string_util.h"

CMediaSessionLive::CMediaSessionLive(const string& session_id, EnMediaSessionType session_type, const SDeviceChannel& dc)
    : CMediaSessionBase(session_id, session_type, dc)
{
}

CMediaSessionLive::~CMediaSessionLive()
{
}

void CMediaSessionLive::Update()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    if ( !IsRunning() )
    {
        return;
    }

    HandlePuUpdate();
    HandleCuUpdate();
    if( CanClose() )
    {
        Warn( "session(%s) can close!\n", session_id_.c_str());
        Stop();
        return;
    }

    if(cloud_storage_agent_)
    {
        cloud_storage_agent_->Update();
    }
}

bool CMediaSessionLive::Start()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    running_ = true;
    start_tick_ = get_current_tick();
    last_active_tick_ = get_current_tick();
    wait_cu_conn_tick_ = get_current_tick();
    wait_pu_conn_tick_ = get_current_tick();

    Warn( "session(%s) start...!\n", session_id_.c_str());
    return true;
}

bool CMediaSessionLive::Stop()
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if(!running_)
        {
            break;
        }

        running_ = false;

        if ( pu_stream_ )
        {
            pu_stream_->Close();
        }

        map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams_.begin();
        while ( it != hi_cu_streams_.end() )
        {
            CCuStream_ptr pCuSteram = it->second;
            pCuSteram->Close();
            ++it;
        }

        if(rtmp_live_agent_)
        {
            rtmp_live_agent_->Stop();
            rtmp_live_agent_.reset();
        }

        Warn( "session(%s) stop!\n", session_id_.c_str());

    } while (0);
    
    return true;
}

bool CMediaSessionLive::OnTcpClose(const CHostInfo& hiRemote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if(pu_stream_ && hiRemote==pu_stream_->GetRemote())
    {
        pu_stream_->OnTcpClose();
        pu_stream_.reset();
        wait_pu_conn_tick_ = get_current_tick(); //开始等待设备建立媒体连接
        Warn("from(%s), pu stream tcp close!", hiRemote.GetNodeString().c_str());
        return true;
    }

    CCuStream_ptr pCuStream = GetCuStream(hiRemote);
    if(pCuStream)
    {
        pCuStream->OnTcpClose();
        RemoveCuStream(hiRemote);
        if(GetCuStreamNum() == 0)
        {
            wait_cu_conn_tick_ = get_current_tick(); //开始等待客户端建立媒体连接
        }
        Warn("from(%s), cu stream tcp close, cu_stream_num(%d)!", hiRemote.GetNodeString().c_str(), GetCuStreamNum());
        return true;
    }

    return false;
}

int CMediaSessionLive::SetCloudStorage(bool onoff)
{
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if(!running_)
        {
            Error( "session(%s) not running, set cloud storage(%s) fail!\n", 
                session_id_.c_str(),
                onoff?"on":"off");
            break;
        }

        if(!pu_stream_ || !pu_stream_->IsConnected())
        {
            Error( "session(%s) set cloud storage(%s) fail, pu stream is not connected!\n", 
                session_id_.c_str(),
                onoff?"on":"off");
            break;
        }

        if(onoff)
        {
            if(!cloud_storage_agent_)
            {
                cloud_storage_agent_ = CCloudStorageAgent_ptr(new CCloudStorageAgent(CServerLogical::GetLogical()->GetServCfg()->GetRecordServ()));
                if(!cloud_storage_agent_)
                {
                    break;
                }

                cloud_storage_agent_->SetDC(session_dc_);

                protocol::VideoCodecInfo video_info;
                pu_stream_->GetVideoInfo(video_info);
                cloud_storage_agent_->SetVideoInfo(video_info);

                protocol::AudioCodecInfo audio_info;
                pu_stream_->GetAudioInfo(audio_info);
                cloud_storage_agent_->SetAudioInfo(audio_info);
            }
        }
        else
        {
            if(cloud_storage_agent_)
            {
                cloud_storage_agent_.reset();
            }
        }
        return 0;
    } while (0);
    return -1;
}

int CMediaSessionLive::SetRtmpLiveAgent(bool onoff)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if(!running_)
        {
            Error( "session(%s) not running, set rtmp live agent(%s) fail!\n", 
                session_id_.c_str(),
                onoff?"on":"off");
            break;
        }

        if(!pu_stream_ || !pu_stream_->IsConnected())
        {
            Error( "session(%s) set rtmp live agent(%s) fail, pu stream is not connected!\n", 
                session_id_.c_str(),
                onoff?"on":"off");
            break;
        }

        if(onoff)
        {
            // send play to device
            //EnPuStreamStatus pu_status = pu_stream_->GetStatus();
            //if(pu_status == en_pu_stream_pause || pu_status==en_pu_stream_connected)
            {
                pu_stream_->Play();
            }

            // open rtmp live agent 
            if(!rtmp_live_agent_)
            {
                /*
                CHostInfo hi_rtmp_pub(
                    CServerLogical::GetLogical()->GetServCfg()->GetRtmpHost(), 
                    CServerLogical::GetLogical()->GetServCfg()->GetRtmpPort() );
                if( !hi_rtmp_pub.IsValid() )
                {
                    Error( "session(%s) set rtmp live agent(%s) fail, get rtmp publish host failed!\n", 
                        session_id_.c_str(),
                        onoff?"on":"off");
                    break;
                }
                string rtmp_url = "rtmp://" + hi_rtmp_pub.GetNodeString() + "/live/" + session_dc_.GetString();
                */

                char szRtmpPubHost[512] = {0};
                int len = snprintf(szRtmpPubHost, sizeof(szRtmpPubHost)-1, "%s:%u",
                            CServerLogical::GetLogical()->GetServCfg()->GetRtmpHost().c_str(),
                            CServerLogical::GetLogical()->GetServCfg()->GetRtmpPort() );
                if( len <=0 )
                {
                    Error( "session(%s) gen rtmp publish host fail!\n", 
                        session_id_.c_str() );
                    break;
                }
                szRtmpPubHost[len] = '\0';

                string rtmp_pub_path = szRtmpPubHost;
                if( !CServerLogical::GetLogical()->GetServCfg()->GetRtmpPath().empty() )
                {
                    rtmp_pub_path += "/" + CServerLogical::GetLogical()->GetServCfg()->GetRtmpPath();
                }

                string rtmp_url = "rtmp://" + rtmp_pub_path + "/live/" + session_dc_.GetString();

                rtmp_live_agent_ = CRtmpLive_ptr(new CRtmpLive());
                if( !rtmp_live_agent_  )
                {
                    Error( "session(%s) create rtmp live agent start fail, rtmp_url=%s!\n", 
                        session_id_.c_str(), rtmp_url.c_str() );
                    break;
                }

                protocol::AudioCodecInfo audio_info;
                if( pu_stream_->GetAudioInfo(audio_info) == 0 )
                {
                    MI_AudioInfo mi_audio_info;
                    mi_audio_info.codec_fmt = audio_info.codec_fmt;
                    mi_audio_info.sample = audio_info.sample;
                    mi_audio_info.channel = audio_info.channel;
                    mi_audio_info.bitwidth = audio_info.bitwidth;
                    mi_audio_info.sepc_size = audio_info.sepc_size;
                    mi_audio_info.sepc_data = audio_info.sepc_data;
                    rtmp_live_agent_->SetAudioInfo(mi_audio_info);
                }

                int ret = rtmp_live_agent_->Start( rtmp_url.c_str() );
                if( ret < 0 )
                {
                    rtmp_live_agent_.reset();
                    Error( "session(%s) rtmp live agent start fail, ret=%d, rtmp_url=%s!\n", 
                        session_id_.c_str(), ret, rtmp_url.c_str() );
                    break;
                }

                Info( "session(%s) open rtmp live agent, rmtp_url=%s\n", session_id_.c_str(), rtmp_url.c_str() );
            }
        }
        else if( rtmp_live_agent_ )
        {
            rtmp_live_agent_->Stop();
            rtmp_live_agent_.reset();
            Info( "session(%s) close rtmp live agent, dc=%s\n", session_id_.c_str(), session_dc_.GetString().c_str() );
        }
        return 0;
    } while (0);
    return -1;
}

bool CMediaSessionLive::OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaConnectReq& req)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    bool ret = false;
    /*
    if (!req.mask&0x01)
    {
        Error( "from(%s), no base session info!", hiRemote.GetNodeString().c_str());
        return false;
    }*/

    StreamMediaConnectResp resp;
    resp.mask = 0x01;
    resp.resp_code = EN_SUCCESS;
    resp.session_id = req.session_id;

    EndPointType endpoint_type = (EndPointType)req.endpoint_type;
    switch(endpoint_type)
    {
    case EP_DEV:
        {
            ret = OnDeviceConnect(sink, hiRemote, req, resp);
            if(ret)
            {
                wait_pu_conn_tick_ = 0;
                if(GetCuStreamNum() == 0)
                {
                    wait_cu_conn_tick_ = get_current_tick();
                }
            }
        }
        break;
    case EP_CU:
    case EP_STREAM:
        {
            ret = OnUserConnect(sink, hiRemote, req, resp);
            if(ret)
            {
                wait_cu_conn_tick_ = 0;
                if(!pu_stream_)
                {
                    wait_pu_conn_tick_ = get_current_tick();
                }
            }
        }
        break;
    default:
        {
            resp.resp_code = EN_ERR_ENDPOINT_UNKWON;
        }
        break;
    }

    MsgHeader resp_header;
    resp_header.msg_id = header.msg_id;
    resp_header.msg_type = MSG_TYPE_RESP;
    resp_header.msg_seq = header.msg_seq;

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << resp_header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();
    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());

    /*if (resp.resp_code == EN_SUCCESS)
    {
        Start();
    }*/
    return ret;
}

bool CMediaSessionLive::OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaDisconnectReq& req)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    int ret = false;

    StreamMediaDisconnectResp resp;
    resp.mask = 0x01;
    resp.session_id = req.session_id;
    resp.resp_code = EN_SUCCESS;

    EndPointType endpoint_type = GetEndpointType(hiRemote);
    switch(endpoint_type)
    {
    case EP_DEV:
        {
            ret = OnDeviceDisconnect(sink, hiRemote, req, resp);
        }
        break;
    case EP_CU:
    case EP_STREAM:
        {
            ret =OnUserDisconnect(sink, hiRemote, req, resp);
        }
        break;
    default:
        {
            resp.resp_code = EN_ERR_ENDPOINT_UNKWON;
        }
        break;
    }
    
    MsgHeader resp_header;
    resp_header.msg_id = header.msg_id;
    resp_header.msg_type = MSG_TYPE_RESP;
    resp_header.msg_seq = header.msg_seq;

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << resp_header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();

    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());
    
    return ret;
}

bool CMediaSessionLive::OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaStatusReq& req)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    bool ret = false;

    StreamMediaStatusResp resp;
    resp.mask = 0x01;
    resp.session_id = req.session_id;
    resp.resp_code = EN_SUCCESS;

    EndPointType endpoint_type = GetEndpointType(hiRemote);
    switch(endpoint_type)
    {
    case EP_DEV:
        {
            ret = OnDeviceStatus(sink, hiRemote, req, resp);
        }
        break;
    case EP_CU:
    case EP_STREAM:
        {
            ret = OnUserStatus(sink, hiRemote, req, resp);
            if( !ret )
            {
                break;
            }
        }
        break;
    default:
        {
            resp.resp_code = EN_ERR_ENDPOINT_UNKWON;
        }
        break;
    }

    MsgHeader resp_header;
    resp_header.msg_id = header.msg_id;
    resp_header.msg_type = MSG_TYPE_RESP;
    resp_header.msg_seq = header.msg_seq;

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << resp_header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();
    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());

    Trace( "from(%s), send media status resp, session_id(%s), resp_code(%d), msg_seq(%u)!\n", 
        hiRemote.GetNodeString().c_str(), resp.session_id.c_str(), resp.resp_code, resp_header.msg_seq );

    return ret;
}

bool CMediaSessionLive::OnPlayReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPlayReq& req)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    bool ret = false;

    StreamMediaPlayResp resp;
    resp.mask = 0x01;
    resp.session_id = req.session_id;
    resp.resp_code = EN_SUCCESS;

    do
    {
        CCuStream_ptr pCuStream = GetCuStream(hiRemote);
        if (!pCuStream)
        {
            resp.resp_code = EN_CU_ERR_STREAM_DISCONNECT;
            break;
        }

        if( ret = pCuStream->OnPlayReq(sink, hiRemote, req, resp) )
        {
            if(!pu_stream_)
            {
                break;
            }
            //EnPuStreamStatus pu_status = pu_stream_->GetStatus();
            //if(pu_status == en_pu_stream_pause || pu_status==en_pu_stream_connected)
            {
                pu_stream_->Play();
            }
        }

    } while (false);

    MsgHeader resp_header;
    resp_header.msg_id = header.msg_id;
    resp_header.msg_type = MSG_TYPE_RESP;
    resp_header.msg_seq = header.msg_seq;

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << resp_header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();

    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());

    return ret;
}

bool CMediaSessionLive::OnPauseReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPauseReq& req)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    bool ret = false;

    StreamMediaPauseResp resp;
    resp.mask = 0x01;
    resp.session_id = req.session_id;
    resp.resp_code = EN_SUCCESS;

    do
    {
        CCuStream_ptr pCuStream = GetCuStream(hiRemote);
        if (!pCuStream)
        {
            resp.resp_code = EN_CU_ERR_STREAM_DISCONNECT;
            break;
        }

        ret = pCuStream->OnPauseReq(sink, hiRemote, req, resp);

        if ( !ret )
        {
            break;
        }

        if ( !pu_stream_ )
        {
            break;
        }

        if( !pu_stream_->IsAlive() )
        {
            pu_stream_->Pause();
        }

    } while (false);

    MsgHeader resp_header;
    resp_header.msg_id = header.msg_id;
    resp_header.msg_type = MSG_TYPE_RESP;
    resp_header.msg_seq = header.msg_seq;

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << resp_header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();

    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());

    return ret;
}

bool CMediaSessionLive::OnCmdReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaCmdReq& req)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    bool ret = false;

    StreamMediaCmdResp resp;
    resp.mask = 0x01;
    resp.session_id = req.session_id;
    resp.resp_code = EN_SUCCESS;

    do 
    {
        CCuStream_ptr pCuStream = GetCuStream(hiRemote);
        if ( !pCuStream )
        {
            resp.resp_code = EN_CU_ERR_STREAM_DISCONNECT;
            break;
        }

        ret = pCuStream->OnCmdReq(sink, hiRemote, req, resp);
        if ( !ret )
        {
            break;
        }

        if ( !pu_stream_ )
        {
            break;
        }

        MediaCMDType cmd_type = (MediaCMDType)req.cmd_type;
        if ( cmd_type == MEDIA_CMD_AUDIO_OPEN )
        {
            //检查pu stream音频是否已打开
            if ( !pu_stream_->IsAudioOpen() )
            {
                pu_stream_->AudioCtrl(true);
            }
            
        }
        else if ( cmd_type == MEDIA_CMD_AUDIO_CLOSE )
        {
            //检查是否可以关闭音频
            if ( !IsAudioCloseEnable() )
            {
                break;
            }

            //检查pu stream音频是否已打开
            if ( !pu_stream_->IsAudioOpen() )
            {
                break;
            }
            pu_stream_->AudioCtrl(false);
        }
        else if ( cmd_type == MEDIA_CMD_VIDEO_OPEN )
        {
            //检查pu stream视频是否已打开
            if ( !pu_stream_->IsVideoOpen() )
            {
                pu_stream_->VideoCtrl(true);
            }
        }
        else if ( cmd_type == MEDIA_CMD_VIDEO_CLOSE )
        {
            //检查pu stream视频是否已打开
            if ( !pu_stream_->IsVideoOpen() )
            {
                pu_stream_->VideoCtrl(false);
            }
        }

    } while (false);

    MsgHeader resp_header;
    resp_header.msg_id = header.msg_id;
    resp_header.msg_type = MSG_TYPE_RESP;
    resp_header.msg_seq = header.msg_seq;

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << resp_header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();

    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());

    return ret;
}

bool CMediaSessionLive::OnPlayResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    if (!pu_stream_ || hiRemote != pu_stream_->GetRemote())
    {
        return false;
    }

    return pu_stream_->OnPlayResp(sink, hiRemote, resp);
}

bool CMediaSessionLive::OnPauseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    if (!pu_stream_ 
        || hiRemote != pu_stream_->GetRemote())
    {
        return false;
    }

    return pu_stream_->OnPauseResp(sink, hiRemote, resp);
}

bool CMediaSessionLive::OnMediaCmdResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    if (!pu_stream_ 
        || hiRemote != pu_stream_->GetRemote())
    {
        return false;
    }

    return pu_stream_->OnCmdResp(sink, hiRemote, resp);
}

bool CMediaSessionLive::OnCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    bool ret = false;

	EndPointType endpoint_type = GetEndpointType(hiRemote);
    switch(endpoint_type)
    {
    case EP_DEV:
        {
            ret=OnDeviceCloseResp(sink, hiRemote, resp);
        }
        break;
    case EP_CU:
    case EP_STREAM:
        {
            ret=OnUserCloseResp(sink, hiRemote, resp);
        }
        break;
    default:
        {
        }
        break;
    }

    return ret;
}

bool CMediaSessionLive::OnFrameNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaFrameNotify& notify)
{
    CPuStream_ptr pu_stream;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!pu_stream_ 
            || hiRemote != pu_stream_->GetRemote())
        {
            return false;
        }
        pu_stream = pu_stream_;
    }
    
    if(!pu_stream->OnFrameNotify(sink, hiRemote, notify))
    {
        return false;
    }

    MI_FrameData_ptr frame_data_ptr;
    if (pu_stream->GetFrameData(notify.frame_seq, notify.frame_type, frame_data_ptr))
    {
        return OnStream(frame_data_ptr);
    }

    return true;
}

bool CMediaSessionLive::OnDeviceConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (pu_stream_)
        {

            resp.resp_code = EN_DEV_ERR_CONNECT_STREAM_REPEAT;
            Error( "from(%s), session_id(%s), session_type(%d), dc(%s), pu stream has already connected!", 
                hiRemote.GetNodeString().c_str(), 
                session_id_.c_str(),
                session_type_,
                session_dc_.GetString().c_str());
            break;
        }

        CPuStream_ptr pPuStream = CPuStream_ptr(new CPuStream());
        if (!pPuStream)
        {
            resp.resp_code = EN_DEV_ERR_CREATE_STREAM_FAIL;
            Error( "from(%s), session_id(%s), session_type(%d), dc(%s), create pu stream failed!", 
                hiRemote.GetNodeString().c_str(), 
                session_id_.c_str(),
                session_type_,
                session_dc_.GetString().c_str() );
            break;
        }

        if( !pPuStream->OnConnect(sink, hiRemote, req, resp) )
        {
            Error( "from(%s), session_id(%s), session_type(%d), dc(%s), handle pu stream connect failed!", 
                hiRemote.GetNodeString().c_str(), 
                session_id_.c_str(),
                session_type_,
                session_dc_.GetString().c_str() );
            break;
        }

        pu_stream_ = pPuStream;

        Debug( "from(%s), session_id(%s), session_type(%d), dc(%s), handle pu stream connect success!", 
            hiRemote.GetNodeString().c_str(), 
            session_id_.c_str(),
            session_type_,
            session_dc_.GetString().c_str() );

		OnStreamStatusReport(SDeviceStreamStatus::enm_dev_media_connected,hiRemote);
        return true;
    } while (false);
    return false;
}

bool CMediaSessionLive::OnUserConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaConnectReq& req, StreamMediaConnectResp& resp)
{
    do 
    {
        //检查连接是否已登录
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams_.find(hiRemote);
            if( it != hi_cu_streams_.end() )
            {
                resp.resp_code = EN_CU_ERR_CONNECT_STREAM_REPEAT;
                Error( "from(%s), session_id(%s), session_type(%d), dc(%s), user_name(%s),cu stream has already connected!", 
                    hiRemote.GetNodeString().c_str(), 
                    session_id_.c_str(),
                    session_type_,
                    session_dc_.GetString().c_str(),
                    req.endpoint_name.c_str() );
                break;
            }
        }

        CCuStream_ptr pCuStream = CCuStream_ptr(new CCuStream());
        if (!pCuStream)
        {
            resp.resp_code = EN_CU_ERR_CREATE_STREAM_FAIL;
            break;
        }

        if (!pCuStream->OnConnect(sink, hiRemote, req, resp))
        {
            break;
        }

        AddCuStream(pCuStream);

        Debug( "from(%s), session_id(%s), session_type(%d), dc(%s), user_name(%s), handle cu stream connect success!", 
            hiRemote.GetNodeString().c_str(), 
            session_id_.c_str(),
            session_type_,
            session_dc_.GetString().c_str(),
            req.endpoint_name.c_str() );

        return true;
    } while (false);
    return false;
}

bool CMediaSessionLive::OnDeviceDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!pu_stream_)
        {
            resp.resp_code = EN_DEV_ERR_STREAM_DISCONNECT;
            break;
        }

        if (hiRemote != pu_stream_->GetRemote() )
        {
            resp.resp_code = EN_DEV_ERR_STREAM_DIFFERENT;
            break;
        }

        pu_stream_->OnDisconnect(sink, hiRemote, req, resp);
        pu_stream_.reset();

		OnStreamStatusReport(SDeviceStreamStatus::enm_dev_media_disconnect,hiRemote);

		return true;
    } while (false);
    return false;
}

bool CMediaSessionLive::OnUserDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaDisconnectReq& req, StreamMediaDisconnectResp& resp)
{
    do
    {
        CCuStream_ptr pCuStream = GetCuStream(hiRemote);
        if (!pCuStream)
        {
            resp.resp_code = EN_CU_ERR_STREAM_DISCONNECT;
            break;
        }

        pCuStream->OnDisconnect(sink, hiRemote, req, resp);

        RemoveCuStream(hiRemote);

        return true;
    } while (false);
    return false;
}

bool CMediaSessionLive::OnDeviceStatus(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!pu_stream_)
        {
            resp.resp_code = EN_DEV_ERR_STREAM_DISCONNECT;
            break;
        }

        if (hiRemote != pu_stream_->GetRemote() )
        {
            resp.resp_code = EN_DEV_ERR_STREAM_DIFFERENT;
            break;
        }

        return pu_stream_->OnStatusReport(sink, hiRemote, req, resp);
    } while (false);
    return false;
}

bool CMediaSessionLive::OnUserStatus(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaStatusReq& req, StreamMediaStatusResp& resp)
{
    do 
    {
        CCuStream_ptr pCuStream = GetCuStream(hiRemote);
        if (!pCuStream)
        {
            resp.resp_code = EN_CU_ERR_STREAM_DISCONNECT;
            Error( "from(%s), session_id(%s), find cu stream faile!\n", 
                hiRemote.GetNodeString().c_str(), req.session_id.c_str() );
            break;
        }

        return pCuStream->OnStatusReport(sink, hiRemote, req, resp);
    } while (false);
    return false;
}

bool CMediaSessionLive::OnDeviceCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp)
{
    do 
    {

		OnStreamStatusReport(SDeviceStreamStatus::enm_dev_media_closed,hiRemote);
        return true;
    } while (false);
    return false;
}

bool CMediaSessionLive::OnUserCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp)
{
    do 
    {

        return true;
    } while (false);
    return false;
}

void CMediaSessionLive::OnStreamStatusReport(int iStatusType, const CHostInfo& hiRemote)
{
	SDeviceStreamStatus device_stream_status;
	{
		boost::lock_guard<boost::recursive_mutex> lock(lock_);
		device_stream_status.mask = 0x01;
		device_stream_status.status = iStatusType;

		if (pu_stream_)
		{
			device_stream_status.did			= pu_stream_->GetDeviceChannel().device_id_;
			device_stream_status.channel_id	    = pu_stream_->GetDeviceChannel().channel_id_;
			device_stream_status.stream_id		= pu_stream_->GetDeviceChannel().stream_id_;
		}

        hiRemote.GetIP(device_stream_status.stream_serv_addr.ip);
		device_stream_status.stream_serv_addr.port	= hiRemote.GetPort();

		struct timeval cur_tv;
		gettimeofday(&cur_tv, NULL);
		device_stream_status.timestamp.tv_sec	= (uint64)cur_tv.tv_sec;
		device_stream_status.timestamp.tv_usec	= (uint64)cur_tv.tv_usec;
	}

	CServerLogical::GetLogical()->GetStreamStatusReport()->AddNotifyStatus(device_stream_status);
}

bool CMediaSessionLive::IsAudioCloseEnable()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    bool enable = true;
    map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams_.begin();
    for (; it != hi_cu_streams_.end(); it++)
    {
        if ( it->second->IsAudioOpen() )
        {
            enable = false;
            break;
        }
    }
    return enable;
}

bool CMediaSessionLive::IsVideoCloseEnable()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    bool enable = true;
    map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams_.begin();
    for (; it != hi_cu_streams_.end(); it++)
    {
        if ( it->second->IsVideoOpen() )
        {
            enable = false;
            break;
        }
    }
    return enable;
}

bool CMediaSessionLive::IsPlayEnable()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if( rtmp_live_agent_ )
    {
        return true;
    }

    bool enable = false;
    map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams_.begin();
    for (; it != hi_cu_streams_.end(); it++)
    {
        EnCuStreamStatus status = it->second->GetStatus();
        if ( status == en_cu_stream_connected || status == en_cu_stream_play )
        {
            enable = true;
            break;
        }
    }
    return enable;
}

int CMediaSessionLive::GetCuStreamNum()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return hi_cu_streams_.size();
}

CCuStream_ptr CMediaSessionLive::GetCuStream(const CHostInfo& hiRemote)
{
    CCuStream_ptr pStream;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams_.find(hiRemote);
        if( it != hi_cu_streams_.end() )
        {
            pStream = it->second;
        }
    }
    return pStream;
}

void CMediaSessionLive::AddCuStream(CCuStream_ptr pStream)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    CHostInfo hiRemote = pStream->GetRemote();
    string strUserName = pStream->GetUserName();
    hi_cu_streams_[hiRemote] = pStream;
    cu_streams_[strUserName].insert(pStream);

    Debug( "session_id(%s), add cu_stream(%s), cu_stream_list_size(%u)!\n", 
        session_id_.c_str(), pStream->GetRemote().GetNodeString().c_str(), cu_streams_.size());
}

void CMediaSessionLive::RemoveCuStream(const CHostInfo& hiRemote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    map<CHostInfo, CCuStream_ptr>::iterator it1 = hi_cu_streams_.find(hiRemote);
    if (it1 == hi_cu_streams_.end())
    {
        Error( "session_id(%s), cannot find cu stream(%s)!\n", 
            session_id_.c_str(), hiRemote.GetNodeString().c_str() );
        return;
    }

    CCuStream_ptr pStream = it1->second;
    map<string, set<CCuStream_ptr> >::iterator it2 = cu_streams_.begin();
    while(it2 != cu_streams_.end())
    {
        set<CCuStream_ptr>& set_contexts = it2->second;
        set_contexts.erase(pStream);
        if (set_contexts.empty())
        {
            cu_streams_.erase(it2++);
        }
        else
        {
            it2++;
        }
    }

    hi_cu_streams_.erase(it1);

    Debug( "session_id(%s), remove cu_stream(%s), cu_stream_list_size(%u)!\n", 
        session_id_.c_str(), hiRemote.GetNodeString().c_str(), cu_streams_.size());
}

EndPointType CMediaSessionLive::GetEndpointType(const CHostInfo& hiRemote)
{
    EndPointType ep_type = protocol::EP_UNKNOWN;
    do 
    {
        //Is from pu stream?
        if ( pu_stream_ && hiRemote == pu_stream_->GetRemote() )
        {
            ep_type = protocol::EP_DEV;
            break;
        }

        //Is from cu stream?
        CCuStream_ptr pCuStream = GetCuStream(hiRemote);
        if ( pCuStream )
        {
            ep_type = protocol::EP_CU;
            break;
        }

        //other

    } while (0);

    return ep_type;
}

bool CMediaSessionLive::OnStream(MI_FrameData_ptr frame_data)
{
    map<CHostInfo, CCuStream_ptr> hi_cu_streams;
    CRtmpLive_ptr rtmp_live_agent;
    CCloudStorageAgent_ptr cloud_storage_agent;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        hi_cu_streams = hi_cu_streams_;
        rtmp_live_agent = rtmp_live_agent_;
        cloud_storage_agent = cloud_storage_agent_;
    }
    
    map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams.begin();
    for ( ; it != hi_cu_streams.end(); ++it )
    {
        CCuStream_ptr pCuStream = it->second;
        pCuStream->OnStream(frame_data);
    }

    if ( rtmp_live_agent )
    {
        rtmp_live_agent->OnStream(frame_data);
    }

    if ( cloud_storage_agent )
    {
        cloud_storage_agent->OnStream(frame_data);
    }

    return true;
}

void CMediaSessionLive::HandlePuUpdate()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (!pu_stream_)
    {
        return;
    }

    pu_stream_->Update();
    if ( !pu_stream_->IsAlive() )
    {
        return;
    }

    EnPuStreamStatus status = pu_stream_->GetStatus();
    switch(status)
    {
    case en_pu_stream_connected:
        {
            if ( IsPlayEnable() )
            {
                pu_stream_->Play();
            }
        }
        break;
    case en_pu_stream_pause:
        {
            if ( IsPlayEnable() )
            {
                pu_stream_->Play();
            }
        }
        break;
    case en_pu_stream_play:
        {
            //检查是否可以控制音频
            bool audio_onoff = !IsAudioCloseEnable();
            if ( audio_onoff != pu_stream_->IsAudioOpen() )
            {
                pu_stream_->AudioCtrl(audio_onoff);
            }
        }
        break;
    default:
        break;
    }
}

void CMediaSessionLive::HandleCuUpdate()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    map<CHostInfo, CCuStream_ptr>::iterator it = hi_cu_streams_.begin();
    while ( it != hi_cu_streams_.end() )
    {
        CCuStream_ptr pCuSteram = it->second;
        pCuSteram->Update();
        if ( !pCuSteram->IsAlive() )
        {
            hi_cu_streams_.erase(it++);
        }
        else
        {
            ++it;
        }
    }
}

bool CMediaSessionLive::CanClose()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (!pu_stream_)
    {
        if(wait_pu_conn_tick_==0)
        {
            return true;
        }
        else if(get_current_tick()-wait_pu_conn_tick_>5*1000)
        {
            return true;
        }
    }
    else if ( !pu_stream_->IsAlive() )
    {
        return true;
    }

    if(cloud_storage_agent_)
    {
        return false;
    }

    if ( rtmp_live_agent_ )
    {
        return false;
    }

    if ( !hi_cu_streams_.empty() )
    {
        return false;
    }
    else
    {
        if(wait_cu_conn_tick_==0)
        {
            return true;
        }
        else if(get_current_tick()-wait_cu_conn_tick_>5*1000)
        {
            return true;
        }
    }

    return false;
}

void CMediaSessionLive::DumpInfo(Variant& info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    info["session_id"] = session_id_;
    info["session_type"] = "live";
    info["duration"] = calc_time_unit((uint32)(get_current_tick() - start_tick_)/1000);
    
    if(pu_stream_)
    {
        Variant pu_info;
        pu_stream_->DumpInfo(pu_info);
        info["pu_stream"] = pu_info;
    }
    
    if(rtmp_live_agent_)
    {
        Variant rtmp_agent;
        rtmp_agent["rtmp_url"] = rtmp_live_agent_->GetRtmpUrl();
        info["rtmp_agent"] = rtmp_agent;
    }
}
