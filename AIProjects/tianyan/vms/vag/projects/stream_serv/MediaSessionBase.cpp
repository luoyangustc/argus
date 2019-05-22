#include "MediaSessionBase.h"

CMediaSessionBase::CMediaSessionBase(const string& session_id, EnMediaSessionType session_type, const SDeviceChannel& dc)
    : session_id_(session_id)
    , session_type_(session_type)
    , session_dc_(dc)
{

}

CMediaSessionBase::~CMediaSessionBase()
{

}

bool CMediaSessionBase::IsRunning()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return running_;
}

string CMediaSessionBase::GetSessionTypeString()
{
    string strSessionType;
    switch(session_type_)
    {
    case MEDIA_SESSION_TYPE_LIVE:               // 实时浏览
        {
            strSessionType = "MEDIA_SESSION_TYPE_LIVE";
        }
        break;
    case MEDIA_SESSION_TYPE_PU_PLAYBACK:        // 前端(NVR/TF卡)录像回放
        {
            strSessionType = "MEDIA_SESSION_TYPE_PU_PLAYBACK";
        }
        break;
    case MEDIA_SESSION_TYPE_PU_DOWNLOAD:        // 前端(NVR/TF卡)录像下载
        {
            strSessionType = "MEDIA_SESSION_TYPE_PU_DOWNLOAD";
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_LIVE:        // 直连实时浏览
        {
            strSessionType = "MEDIA_SESSION_TYPE_DIRECT_LIVE";
        }
        break;
    case MEDIA_SESSION_TYPE_DIRECT_PU_PLAYBACK: // 直连前端(NVR/TF卡)录像回放
        {
            strSessionType = "MEDIA_SESSION_TYPE_DIRECT_PU_PLAYBACK";
        }
        break;  
    case MEDIA_SESSION_TYPE_DIRECT_PU_DOWNLOAD: // 直连前端(NVR/TF卡)录像下载
        {
            strSessionType = "MEDIA_SESSION_TYPE_DIRECT_PU_DOWNLOAD";
        }
        break;
    default:
        {
            strSessionType = "MEDIA_SESSION_TYPE_UNKNOWN";
        }
        break; 
    }
    return strSessionType;
}