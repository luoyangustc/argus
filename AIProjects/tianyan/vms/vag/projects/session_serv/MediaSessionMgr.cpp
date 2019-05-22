#include "MediaSessionMgr.h"
#include "ServerLogical.h"
#include "base/include/logging_posix.h"

void CMediaSessionMgr::Update()
{
    map<string, MediaSession_ptr> sessions;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        sessions = sessions_;
    }

    map<string, MediaSession_ptr>::iterator it = sessions.begin();
    for ( ; it != sessions.end(); ++it)
    {
        MediaSession_ptr pMediaSession = it->second;
        pMediaSession->Update();
        if ( !pMediaSession->IsAlive() )
        {
            RemoveMediaSession( pMediaSession->GetSessionId() );
        }
    }
}

MediaSession_ptr CMediaSessionMgr::GetMediaSession( const SDeviceChannel& dc, SessionType session_type, bool is_create )
{
    MediaSession_ptr pMediaSession;

    switch ( session_type )
    {
    case en_session_type_live:
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<SDeviceChannel, MediaSession_ptr>::iterator it = live_sessions_.find(dc);
            if( it != live_sessions_.end() )
            {
                pMediaSession = it->second;
            }
            else if(is_create)
            {
                pMediaSession = CreateMediaSession( dc, session_type );
            }
        }
        break;
    case en_session_type_playback:
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<SDeviceChannel, MediaSession_ptr>::iterator it = playback_sessions_.find(dc);
            if( it != playback_sessions_.end() )
            {
                pMediaSession = it->second;
            }
            else if(is_create)
            {
                pMediaSession = CreateMediaSession( dc, session_type );
            }
        }
        break;
    default:
        break;
    }

    return pMediaSession;
}

MediaSession_ptr CMediaSessionMgr::GetMediaSession( const string& session_id)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    MediaSession_ptr pSession;

    map<string, MediaSession_ptr>::iterator it = sessions_.find(session_id);
    if( it == sessions_.end())
    {
        return pSession;
    }
    else
    {
        return it->second;
    }
}

MediaSession_ptr CMediaSessionMgr::CreateMediaSession( const SDeviceChannel& dc, SessionType session_type )
{
    MediaSession_ptr pMediaSession;

    if( !GetService()->GetDeviceContext( dc.device_id_ ) )
    {
        Error( "device is not online, create media session fail, dc(%s), session_type(%d),!", 
            dc.GetString().c_str(), session_type );
        return pMediaSession;
    }

    switch( session_type )
    {
    case en_session_type_live:
        {
            pMediaSession = MediaSession_ptr( new MediaSession( dc, session_type ) );
            if( pMediaSession )
            {
                boost::lock_guard<boost::recursive_mutex> lock(lock_);
                live_sessions_[dc] = pMediaSession;
                sessions_[ pMediaSession->GetSessionId() ] = pMediaSession;
                Debug( "create live media session success, session_id(%s), toatal_size(%u)",
                    pMediaSession->GetSessionId().c_str(),
                    live_sessions_.size() );
            }
        }
        break;
    case en_session_type_playback:
        {
            pMediaSession = MediaSession_ptr( new MediaSession( dc, session_type ) ); 
            if( pMediaSession )
            {
                boost::lock_guard<boost::recursive_mutex> lock(lock_);
                playback_sessions_[dc] = pMediaSession;
                sessions_[ pMediaSession->GetSessionId() ] = pMediaSession;
                Debug( "create playback media session success, session_id(%s), toatal_size(%u)",
                    pMediaSession->GetSessionId().c_str(),
                    playback_sessions_.size() );                 
            }
        }
        break;
    default:
        break;
    }

    return pMediaSession;
}

MediaSession_ptr CMediaSessionMgr::RebuildMediaSession( const SDeviceChannel& dc, SessionType session_type, string session_id, HostAddr stream_addr )
{
    MediaSession_ptr pSession;

    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);

        pSession = MediaSession_ptr( new MediaSession(dc, session_type, session_id) );
        if ( !pSession )
        {
            Error("rebuild media session failed, session_id(%s), stream_host(%s:%d)!",
                session_id.c_str(), stream_addr.ip.c_str(), stream_addr.port );
            break;
        }

        pSession->hi_stream_serv_ = CHostInfo(stream_addr.ip, stream_addr.port);
        pSession->UpdateSessionState(en_session_sts_active);

        if( session_type == en_session_type_live )
        {
            live_sessions_[dc] = pSession;
            sessions_[ pSession->GetSessionId() ] = pSession;
        }
        else if( session_type == en_session_type_playback )
        {
            playback_sessions_[dc] = pSession;
            sessions_[ pSession->GetSessionId() ] = pSession;
        }

        Debug( "rebuild media session success, session_id(%s), stream_serv(%s:%d).", 
            session_id.c_str(), stream_addr.ip.c_str(), stream_addr.port );
    } while (0);

    return pSession;
}

void CMediaSessionMgr::RemoveMediaSession(const SDeviceChannel& dc , SessionType session_type)
{
    switch( session_type )
    {
    case en_session_type_live:
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<SDeviceChannel, MediaSession_ptr>::iterator it = live_sessions_.find(dc);
            if( it != live_sessions_.end() )
            {
                it->second->Stop();

                string session_id = it->second->GetSessionId();
                sessions_.erase(session_id);
                live_sessions_.erase(it);
                Debug( "remove live session, session_id(%s), live_session_num(%u)", 
                    session_id.c_str(), live_sessions_.size() );
            }
        }
        break;
    case en_session_type_playback:
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<SDeviceChannel, MediaSession_ptr>::iterator it = playback_sessions_.find(dc);
            if( it != playback_sessions_.end() )
            {
                it->second->Stop();

                string session_id = it->second->GetSessionId();
                sessions_.erase(session_id);
                playback_sessions_.erase(it);
                Debug( "remove playback session, session_id(%s), dc(%s), playback_session_num(%u)", 
                    session_id.c_str(), playback_sessions_.size() );
            }
        }
        break;
    default:
        break;
    }
}



void CMediaSessionMgr::RemoveMediaSession( const string& session_id )
{
    MediaSession_ptr pSession = GetMediaSession( session_id );
    if ( pSession )
    {
        RemoveMediaSession ( pSession->GetDC(), pSession->GetSessionType() );
    }
}

bool CMediaSessionMgr::OnUserMediaOpenReq(const SDeviceChannel& dc, SessionType session_type, const CHostInfo& hiRemote, IMediaSessionSink_ptr user_ctx)
{
    MediaSession_ptr pSession = GetMediaSession(dc, session_type, true);
    if(!pSession)
    {
        return false;
    }
    return pSession->OnUserMediaOpenReq(hiRemote, user_ctx);
}

bool CMediaSessionMgr::OnUserClose( const SDeviceChannel& dc, SessionType session_type, const CHostInfo& hiRemote )
{
    MediaSession_ptr pSession = GetMediaSession( dc, session_type );
    if ( !pSession )
    {
        return false;
    }

    pSession->OnUserClose( hiRemote );

    return true;
}

bool CMediaSessionMgr::OnUserClose( string session_id, const CHostInfo& hiRemote )
{
    MediaSession_ptr pSession = GetMediaSession( session_id );
    if( !pSession )
    {
        return false;
    }

    pSession->OnUserClose( hiRemote );

    return true;
}

bool CMediaSessionMgr::OnDeviceClose( const string& device_id )
{
    vector<MediaSession_ptr> wait_close_sessions;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);

        map<SDeviceChannel, MediaSession_ptr>::iterator it = live_sessions_.begin();
        while( it != live_sessions_.end() )
        {
            if( it->first.device_id_ == device_id )
            {
                wait_close_sessions.push_back(it->second);

                sessions_.erase( it->second->GetSessionId() );
                live_sessions_.erase(it++);
            }
            else
            {
                it++;
            }
        }

        it = playback_sessions_.begin();
        while( it != playback_sessions_.end() )
        {
            if( it->first.device_id_ == device_id )
            {
                wait_close_sessions.push_back(it->second);

                sessions_.erase( it->second->GetSessionId() );
                playback_sessions_.erase(it++);
            }
            else
            {
                it++;
            }
        }
    }
    
    vector<MediaSession_ptr>::iterator it = wait_close_sessions.begin();
    for(; it != wait_close_sessions.end(); ++it)
    {
        (*it)->OnDeviceClose();
    }

    return true;
}

bool CMediaSessionMgr::OnDeviceStatusReport(const vector<DeviceMediaSessionStatus>& status_list)
{
    vector<DeviceMediaSessionStatus>::const_iterator it=status_list.begin();
    for( ; it!=status_list.end(); it++)
    {
        std::string session_id = it->session_id;
        uint8 session_status = it->session_status; // 媒体会话状态 0x00：会话结束 01：会话建立中 0x02：会话ok

        Debug("recv device media session status report, session_status(%d), session_id(%s).", 
            session_status, session_id.c_str() );

        if ( session_status == 0x00 ) //会话结束
        {
            RemoveMediaSession(session_id);
        }
        else
        {
            MediaSession_ptr pSession = GetMediaSession(session_id);
            if ( pSession )
            {
                pSession->OnDeviceStatusReport(*it);
            }
            else
            {
                Warn( "rebuild media session, session_id(%s)!", session_id.c_str() );
                
                SDeviceChannel dc( it->device_id, it->channel_id, it->stream_id );
                RebuildMediaSession( dc, (SessionType)it->session_type, it->session_id, it->stream_addr );
            }
        }
    }
}

bool CMediaSessionMgr::OnDeviceMediaOpenAck(const DeviceMediaOpenResp& resp)
{
    MediaSession_ptr pSession = GetMediaSession(resp.session_id);
    if ( !pSession )
    {
        Error("session_id(%s) not found!", resp.session_id.c_str());
        return false;
    }

    if( !pSession->OnDeviceMediaOpenAck(resp) )
    {
        return false;
    }
}