#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/tuple/tuple.hpp>
#include "MediaSession.h"
#include "base/include/logging_posix.h"
#include "third_party/json/include/json.h"
#include "third_party/json/include/value.h"
#include "ServerLogical.h"
#include "DeviceMgr.h"
#include "DeviceContext.h"

MediaSession::MediaSession( const SDeviceChannel& dc, SessionType session_type, const string& session_id )
    : dc_(dc)
    , session_type_(session_type)
    , session_state_(en_session_sts_init)
    , session_state_tick_(0)
{
    if ( session_id.empty() )
    {
        GenerateSessionId();
    }
    else
    {
        session_id_ = session_id;
    }

    Debug( "session_id(%s),construct...", session_id_.c_str() );
}

MediaSession::~MediaSession()
{
    Debug( "session_id(%s), destroy...",  session_id_.c_str() );
}

CHostInfo MediaSession::GetStreamServer()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return hi_stream_serv_;
}

void MediaSession::GenerateSessionId()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( !session_id_.empty() )
    {
        return;
    }

    session_id_ = dc_.GetString();
    session_id_ += "_";
    session_id_ += boost::lexical_cast<std::string>( session_type_ );
    session_id_ += "_";
    session_id_ += boost::posix_time::to_iso_string( boost::posix_time::second_clock::local_time() );
    session_id_ += "_";
    session_id_ += boost::lexical_cast<std::string>( (unsigned int*)this );
}

const string& MediaSession::GetSessionId()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return session_id_;
}

SessionType MediaSession::GetSessionType() 
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return (SessionType)(int)session_type_; 
}

bool MediaSession::IsAlive()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if( session_state_ == en_session_sts_close )
    {
        return false;
    }

    return true;
}

SDeviceChannel MediaSession::GetDC()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return dc_;
}

int MediaSession::GetSessionState() 
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return session_state_; 
}

void MediaSession::UpdateSessionState( SessionState state )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    session_state_ = state;
    session_state_tick_ = get_current_tick();
}

void MediaSession::Update()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( session_state_ == en_session_sts_get_stream_serv )
    {
        if ( get_current_tick() - session_state_tick_ > 5*1000 )
        {
            Error( "get stream serv timeout, session_id(%s).", session_id_.c_str() );

            this->Stop();
        }
    }
    else if ( session_state_ == en_session_sts_establishing )
    {
        if ( get_current_tick() - session_state_tick_ > 5*1000 )
        {
            Error( "establish session timeout, session_id(%s).", session_id_.c_str() );

            this->Stop();
        }
    }
    else
    {
        ;;
    }
}

bool MediaSession::Start()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    CHostInfo hiNil;
    string url = this->GetStreamSchedulerUrl(false, hiNil);
    if ( url.empty() )
    {
        Error("Get stream scheduler url failed, session_id(%s)", session_id_.c_str() );

        return false;
    }

    MediaSession* user_data = this;

    uint32_t request_id;
    int ret = WebRequest::instance().SubmitHttpRequest( url.c_str(), &request_id, OnHttpStreamScheduleCallback, user_data, NULL, 0, 2, 2 );
    if ( ret != 0 )
    {
        Error("async send http request failed, session_id(%s), url(%s)", session_id_.c_str(), url.c_str() );
        return false;
    }
    else
    {
        Debug("async send http request ok, session_id(%s), request_id(%u), url(%s)", session_id_.c_str(), request_id, url.c_str() );
    }

    UpdateSessionState( en_session_sts_get_stream_serv );

    Debug(" media session start, session_id(%s)", session_id_.c_str() );

    return true;
}

void MediaSession::Stop()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if( session_state_ < en_session_sts_active )
    {
        map<CHostInfo,IMediaSessionSink_ptr>::iterator it = user_ctxs_.begin();
        for( ; it != user_ctxs_.end(); ++it )
        {
            it->second->OnMediaOpenAck( -1, dc_, session_type_, session_id_, hi_stream_serv_);
        }
    }
    
    session_state_ = en_session_sts_close;

    Debug( "media session stop, session_id(%s).", session_id_.c_str() );
}

int MediaSession::OnHttpStreamScheduleCallback( uint32_t request_id, void* user_data, int err_code, int http_code, const char* http_resp )
{
    int ret = -1;
    MediaSession* pSession = (MediaSession*)user_data;
    vector<HostAddr> stream_list;

    Debug( "http stream schedule request callback, session_id(%s), request_id(%u), err_code(%d), http_code(%d), http_resp(%s).",
        pSession->GetSessionId().c_str(), request_id, err_code, http_code, http_resp );

    do
    {
        if ( pSession->GetSessionState() != en_session_sts_get_stream_serv )
        {
            ret = -1;
            break;
        }

        if ( err_code!= 0 )
        {
            ret = -2;
            break;
        }

        if ( http_code < 200 || http_code >= 300 )
        {
            ret = -3;
            break;
        }

        if ( !ParseStreamHostaddr( http_resp, stream_list ) )
        {
            ret = -4;
            break;
        }

        CDeviceMgr_ptr pDevMgr = CServerLogical::GetLogical()->GetDeviceMgr();
        CDeviceContext_ptr pDevCtx = pDevMgr->GetDeviceContext( pSession->dc_.device_id_ );
        if ( !pDevCtx )
        {
            ret = -5;
            break; 
        }

        SMediaDesc desc;
        desc.video_open = true;
        desc.audio_open = true;
        if ( !pDevCtx->MediaOpen( pSession->dc_, pSession->GetSessionId(), (int)pSession->GetSessionType(), desc, stream_list ) )
        {
            Error("session_id(%s), media open failed!", 
                pSession->GetSessionId().c_str(), (int)pSession->GetSessionType(), pSession->dc_.GetString().c_str());
            ret = -6;
            break;
        }

        pSession->UpdateSessionState( en_session_sts_establishing );

        ret = 0;

    }while(0);

    if(ret < 0)
    {
        pSession->Stop();
    }

    return ret;
}

bool MediaSession::ParseStreamHostaddr(const char* szjson, vector<HostAddr>& stream_list)
{
    Json::Reader reader(Json::Features::strictMode());
    Json::Value value;  
    if(!reader.parse(szjson, value))
    {
        return false;
    }

    if (value["code"].isNull() || !value["code"].isInt() || value["code"] != 0)
    {
        return false;
    }

    if ( value["data"].isNull() )
    {
        return false;
    }

    const Json::Value data = value["data"];
    if (data["server"].isNull() || !data["server"].isArray())
    {
        return false;
    }

    const Json::Value servers = data["server"];
    for (unsigned int i = 0; i < servers.size(); i++)
    {
        if (servers[i].isString())
        {
            string ip_port = servers[i].asString();
            std::string::size_type pos = ip_port.find(":");
            if (pos != std::string::npos)
            {
                string ip = ip_port.substr(0,pos);
                uint16 port = atoi(ip_port.substr(pos+1).c_str());
                if (ip.length() > 0 && port > 0)
                {
                    HostAddr ha;
                    ha.ip = ip;
                    ha.port = port;
                    stream_list.push_back(ha);
                }
            }
        }
    }

    return stream_list.size() > 0;
}

std::string MediaSession::GetStreamSchedulerUrl( bool is_relay, CHostInfo hi_client )
{
    CDeviceMgr_ptr pDevMgr = CServerLogical::GetLogical()->GetDeviceMgr();
    CDeviceContext_ptr pDevCtx = pDevMgr->GetDeviceContext( dc_.device_id_ );
    if (!pDevCtx)
    {
        return "";
    }

    std::string client_ip, client_port;
    hi_client.GetNodeString(client_ip, client_port);

    std::string device_ip, device_port;
    pDevCtx->GetRemote().GetNodeString(device_ip, device_port);

    char szUrl[512] ={0};
    int len = snprintf( szUrl, sizeof(szUrl) - 1, 
        "http://%s/streamrrs?device_id=%s&client_real_ip=%s&device_ip=%s&business_channel=stream.gslb.qn.com", 
        CServerLogical::GetLogical()->GetStreamScheduler(), 
        pDevCtx->GetDeviceId().c_str(), 
        client_ip.c_str(),
        device_ip.c_str() );
    if ( len <= 0 )
    {
        return "";
    }

    return szUrl;
}

bool MediaSession::OnUserMediaOpenReq(const CHostInfo& hiRemote, IMediaSessionSink_ptr user_ctx)
{
    bool ret = true;

    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( session_state_ == en_session_sts_init )
    {
        ret = Start();
        user_ctxs_[hiRemote] = user_ctx;
    }
    else if ( session_state_ == en_session_sts_active )
    {
        user_ctx->OnMediaOpenAck( 0, dc_, session_type_, session_id_, hi_stream_serv_);
        user_ctxs_[hiRemote] = user_ctx;
    }
    else if( session_state_ == en_session_sts_close )
    {
        user_ctx->OnMediaOpenAck( -1, dc_, session_type_, session_id_, hi_stream_serv_);
    }

    return ret;
}

bool MediaSession::OnUserClose(const CHostInfo& hiRemote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    user_ctxs_.erase(hiRemote);

    return true;
}

bool MediaSession::OnDeviceClose()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    this->Stop();

    return true;
}

bool MediaSession::OnDeviceStatusReport(const protocol::DeviceMediaSessionStatus& status)
{
    return true;
}

bool MediaSession::OnDeviceMediaOpenAck(const DeviceMediaOpenResp& resp)
{
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);

        if ( session_state_ != en_session_sts_establishing )
        {
            break;
        }

        if ( resp.resp_code == EN_SUCCESS )
        {
            Debug( "open media success, session_id(%s).", session_id_.c_str() );

            UpdateSessionState(en_session_sts_active);
            hi_stream_serv_ = CHostInfo(resp.stream_server.ip, resp.stream_server.port);

            // notify to clients
            map<CHostInfo,IMediaSessionSink_ptr>::iterator it = user_ctxs_.begin();
            for( ; it != user_ctxs_.end(); ++it )
            {
                it->second->OnMediaOpenAck( 0, dc_, session_type_, session_id_, hi_stream_serv_);
            }
        }
        else
        {
            Debug( "open media failed, session_id(%s), dev_resp_code(%d).", session_id_.c_str(), resp.resp_code );

            this->Stop();
        }
    }while(false);

    return true;
}
