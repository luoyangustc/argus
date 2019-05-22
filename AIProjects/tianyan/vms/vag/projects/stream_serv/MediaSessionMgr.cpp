#include "MediaSessionMgr.h"

CMediaSessionMgr::CMediaSessionMgr()
{
}

CMediaSessionMgr::~CMediaSessionMgr()
{
}

void CMediaSessionMgr::Update()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    map<string, CMediaSessionBase_ptr>::iterator it = media_sessions_.begin();
    while ( it != media_sessions_.end() )
    {
        CMediaSessionBase_ptr pSession = it->second;
        (void)pSession->Update();
        if ( !pSession->IsRunning() )
        {
            media_sessions_.erase(it++);
        }
        else
        {
            ++it;
        }
    }
}

bool CMediaSessionMgr::Start()
{

    return true;
}

bool CMediaSessionMgr::Stop()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    
    //stop all sessions
    map<string, CMediaSessionBase_ptr>::iterator it = media_sessions_.begin();
    while ( it != media_sessions_.end() )
    {
        CMediaSessionBase_ptr pSession = it->second;
        pSession->Stop();

        ++it;
    }

    //clear session map
    media_sessions_.clear();
    live_sessions_.clear();

    return true;
}

bool CMediaSessionMgr::OnTCPClosed( const CHostInfo& hiRemote )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    map<string, CMediaSessionBase_ptr>::iterator it = media_sessions_.begin();
    while ( it != media_sessions_.end() )
    {
        CMediaSessionBase_ptr pSession = it->second;
        (void)pSession->OnTcpClose(hiRemote);

        ++it;
    }

    return true;
}

CMediaSessionBase_ptr CMediaSessionMgr::GetMediaSession(string session_id)
{
    CMediaSessionBase_ptr pMediaSession;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        map<string, CMediaSessionBase_ptr>::iterator iter = media_sessions_.find(session_id);
        if (iter != media_sessions_.end())
        {
            pMediaSession = iter->second;
        }
    }
    return pMediaSession;
}

CMediaSessionBase_ptr CMediaSessionMgr::GetMediaSession(EnMediaSessionType session_type, const SDeviceChannel& dc)
{
    CMediaSessionBase_ptr pMediaSession;
    switch(session_type)
    {
    case MEDIA_SESSION_TYPE_LIVE: // 实时浏览
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<SDeviceChannel, CMediaSessionBase_ptr>::iterator iter = live_sessions_.find(dc);
            if (iter != live_sessions_.end())
            {
                pMediaSession = iter->second;
            }
        }
        break;
    case MEDIA_SESSION_TYPE_PU_PLAYBACK: // 前端(NVR/TF卡)录像回放
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_PU_DOWNLOAD: // 前端(NVR/TF卡)录像下载
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_LIVE:  // 直连实时浏览
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_PU_PLAYBACK: // 直连前端(NVR/TF卡)录像回放
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;  
    case MEDIA_SESSION_TYPE_DIRECT_PU_DOWNLOAD: // 直连前端(NVR/TF卡)录像下载
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    default:
        {

        }
        break;
    }
    return pMediaSession;
}

CMediaSessionBase_ptr CMediaSessionMgr::CreateMediaSession(string session_id, EnMediaSessionType session_type, const SDeviceChannel& dc )
{
    CMediaSessionBase_ptr pMediaSession;
    switch(session_type)
    {
    case MEDIA_SESSION_TYPE_LIVE: // 实时浏览
        {
            pMediaSession = CMediaSessionBase_ptr(new CMediaSessionLive(session_id, session_type, dc));
            if ( pMediaSession && pMediaSession->Start() )
            {
                boost::lock_guard<boost::recursive_mutex> lock(lock_);
                live_sessions_[dc] = pMediaSession;
                media_sessions_[session_id] = pMediaSession;
            }
            else
            {
                pMediaSession.reset();
            }
        }
        break;
    case MEDIA_SESSION_TYPE_PU_PLAYBACK: // 前端(NVR/TF卡)录像回放
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_PU_DOWNLOAD: // 前端(NVR/TF卡)录像下载
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_LIVE: // 直连实时浏览
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_PU_PLAYBACK: // 直连前端(NVR/TF卡)录像回放
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;  
    case MEDIA_SESSION_TYPE_DIRECT_PU_DOWNLOAD: // 直连前端(NVR/TF卡)录像下载
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    default:
		{
			Error( "create session failed,session_id(%s), session_type(%d), dc(%s)", session_id.c_str(),session_type,dc.GetString().c_str());
		}
		break;
    }

    return pMediaSession;
}

void CMediaSessionMgr::RemoveMediaSession(const string& session_id)
{
    do 
    {
        CMediaSessionBase_ptr pSession;
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<string, CMediaSessionBase_ptr>::iterator iter = media_sessions_.find(session_id);
            if( iter == media_sessions_.end())
            {
                break;
            }
            pSession = iter->second;
        }

        if(!pSession)
        {
            break;
        }

        RemoveMediaSession(pSession->GetSessionType(), pSession->GetSessionDC());

    } while (0);
    
}

void CMediaSessionMgr::RemoveMediaSession(EnMediaSessionType session_type, const SDeviceChannel& dc)
{
    switch(session_type)
    {
    case MEDIA_SESSION_TYPE_LIVE: // 实时浏览
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<SDeviceChannel, CMediaSessionBase_ptr>::iterator iter = live_sessions_.find(dc);
            if (iter != live_sessions_.end())
            {
                iter->second->Stop();

                string session_id = iter->second->GetSessionID();
                media_sessions_.erase(session_id);
                live_sessions_.erase(iter);
            }
        }
        break;
    case MEDIA_SESSION_TYPE_PU_PLAYBACK: // 前端(NVR/TF卡)录像回放
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_PU_DOWNLOAD: // 前端(NVR/TF卡)录像下载
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_LIVE: // 直连实时浏览
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_PU_PLAYBACK: // 直连前端(NVR/TF卡)录像回放
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;  
    case MEDIA_SESSION_TYPE_DIRECT_PU_DOWNLOAD: // 直连前端(NVR/TF卡)录像下载
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
        }
        break;
    default:
        {

        }
        break; 
    }
}

bool CMediaSessionMgr::OnConnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaConnectReq& req)
{
    do
    {
        string session_id = req.session_id;
        EnMediaSessionType session_type = (EnMediaSessionType)req.session_type;
        SDeviceChannel session_dc(req.device_id, req.channel_id, req.stream_id);

        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            pSession = CreateMediaSession(session_id, session_type, session_dc);
        }
		if (!pSession)
		{
			break;
		}
        return pSession->OnConnect(sink, hiRemote, header, req);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnDisconnect(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaDisconnectReq& req)
{
    do 
    {
        string session_id = req.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnDisconnect(sink, hiRemote, header, req);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaStatusReq& req)
{
    do 
    {
        string session_id = req.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            Error( "from(%s), session_id(%s), find session faile!\n", 
                hiRemote.GetNodeString().c_str(), session_id.c_str() );
            break;
        }
        return pSession->OnStatusReport(sink, hiRemote, header, req);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnPlayReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPlayReq& req)
{
    do 
    {
        string session_id = req.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnPlayReq(sink, hiRemote, header, req);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnPauseReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaPauseReq& req)
{
    do 
    {
        string session_id = req.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnPauseReq(sink, hiRemote, header, req);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnCmdReq(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaCmdReq& req)
{
    do
    {
        string session_id = req.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnCmdReq(sink, hiRemote, header, req);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnPlayResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPlayResp& resp)
{
    do 
    {
        string session_id = resp.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnPlayResp(sink, hiRemote, resp);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnPauseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaPauseResp& resp)
{
    do 
    {
        string session_id = resp.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnPauseResp(sink, hiRemote, resp);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnMediaCmdResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCmdResp& resp)
{
    do 
    {
        string session_id = resp.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnMediaCmdResp(sink, hiRemote, resp);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnCloseResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const StreamMediaCloseResp& resp)
{
    do 
    {
        string session_id = resp.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnCloseResp(sink, hiRemote, resp);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnFrameNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaFrameNotify& notify)
{
    do 
    {
        string session_id = notify.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnFrameNotify(sink, hiRemote, header, notify);
    } while (false);
    return false;
}

bool CMediaSessionMgr::OnEosNotify(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const MsgHeader& header, const StreamMediaEosNotify& notify)
{
    do
    {
        string session_id = notify.session_id;
        CMediaSessionBase_ptr pSession = GetMediaSession(session_id);
        if (!pSession)
        {
            break;
        }
        return pSession->OnEosNotify(sink, hiRemote, header, notify);
    } while (false);
    return false;
}

void CMediaSessionMgr::DumpInfo(Variant& info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    map<string, CMediaSessionBase_ptr>::iterator it = media_sessions_.begin();
    for ( ; it != media_sessions_.end(); ++it )
    {
        CMediaSessionBase_ptr pSession = it->second;
        if( pSession )
        {
            Variant session_info;
            pSession->DumpInfo(session_info);
            info[pSession->GetSessionTypeString()].PushToArray(session_info);
        }
    }
}

void CMediaSessionMgr::DumpInfo(const string& device_id, Variant& info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    map<string, CMediaSessionBase_ptr>::iterator it = media_sessions_.begin();
    for ( ; it != media_sessions_.end(); ++it )
    {
        CMediaSessionBase_ptr pSession = it->second;
        if( pSession )
        {
            SDeviceChannel dc = pSession->GetSessionDC();
            if( dc.device_id_ == device_id )
            {
                Variant session_info;
                pSession->DumpInfo(session_info);
                info[pSession->GetSessionTypeString()].PushToArray(session_info);
            }
        }
    }
}

void CMediaSessionMgr::DumpInfo(const SDeviceChannel& dc, Variant& info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    map<string, CMediaSessionBase_ptr>::iterator it = media_sessions_.begin();
    for ( ; it != media_sessions_.end(); ++it )
    {
        CMediaSessionBase_ptr pSession = it->second;
        if( pSession )
        {
            if( pSession->GetSessionDC() == dc )
            {
                Variant session_info;
                pSession->DumpInfo(session_info);
                info[pSession->GetSessionTypeString()].PushToArray(session_info);
            }
        }
    }
}

