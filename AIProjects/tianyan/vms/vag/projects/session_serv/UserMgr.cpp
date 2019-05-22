#include <stack>
#include "logging_posix.h"
#include "UserMgr.h"
#include "base/include/DeviceID.h"
#include "DeviceMgr.h"
#include "ServerLogical.h"

extern bool g_enable_multiPoint_access;

CUserMgr::CUserMgr()
{
}

CUserMgr::~CUserMgr(void)
{
}

void CUserMgr::Update()
{
    map< CHostInfo, CUserContext_ptr > tmp_user_contexts;
    map<CHostInfo, CWebUserContext_ptr> tmp_webcontexts;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        tmp_user_contexts = user_contexts_;
        tmp_webcontexts = webcontexts_;
    }

    {
        map<CHostInfo, CUserContext_ptr>::iterator it = tmp_user_contexts.begin();
        for (; it != tmp_user_contexts.end(); ++it)
        {
            CUserContext_ptr pContext = it->second;
            pContext->Update();
            if( !pContext->IsAlive() )
            {
                CHostInfo hi_remote = it->first;
                RemoveUserContext(hi_remote);
            }
        }
    }

    {
        map<CHostInfo, CWebUserContext_ptr>::iterator it = tmp_webcontexts.begin();
        for (; it != tmp_webcontexts.end(); ++it)
        {
            CWebUserContext_ptr pWebContext = it->second;
            pWebContext->Update();
            if( !pWebContext->IsAlive() )
            {
                webcontexts_.erase(it->first);
            }
        }
    }
}

void CUserMgr::DoIdleTask()
{

}

bool CUserMgr::OnTCPClosed( const CHostInfo& hiRemote )
{
    bool ret = false;
    do 
    {
        CUserContext_ptr user_context = GetUserContext(hiRemote);
        if( user_context )
        {
            user_context->OnTcpClose(hiRemote);
            string user_name = user_context->GetUserName();
            RemoveUserContext(hiRemote);
            ret = true;
            Debug("(host_info(%s), user_name(%s), closed!", hiRemote.GetNodeString().c_str(), user_name.c_str());
            break;
        }

        map< CHostInfo, CWebUserContext_ptr > webcontexts;
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            webcontexts = webcontexts_;
        }

        {
            map< CHostInfo, CWebUserContext_ptr >::iterator it = webcontexts.begin();
            for ( ; it!=webcontexts.end(); ++it )
            {
                CWebUserContext_ptr pCtx = it->second;
                if ( pCtx->GetRemote() == hiRemote )
                {
                    pCtx->OnTcpClose(hiRemote);
                    ret = true;
                    Debug("(host_info(%s), req_url(%s), closed!", 
                        hiRemote.GetNodeString().c_str(), pCtx->GetReqUrl().c_str());
                    break;
                }
            }
        }

    } while (0);

    return ret;
}

uint32 CUserMgr::GetConnectNum()
{
    return user_contexts_.size();
}

CUserContext_ptr CUserMgr::GetUserContext( const CHostInfo& hiRemote )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    CUserContext_ptr pUserContext;
    map<CHostInfo, CUserContext_ptr>::iterator it = user_contexts_.find(hiRemote);
    if( it != user_contexts_.end() )
    {
        pUserContext = it->second;
    }

    return pUserContext;
}

void CUserMgr::GetUserContextsByUsername( const string& user_name, vector<CUserContext_ptr>& user_contexts )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    map<CHostInfo, CUserContext_ptr >::iterator it = user_contexts_.begin();
    for( ; it != user_contexts_.end(); ++it)
    {
        if( it->second->GetUserName() == user_name )
        {
            user_contexts.push_back(it->second);
        }
    }
}

string CUserMgr::GetUserName( const CHostInfo& hiRemote )
{
    string strName = "";
    CUserContext_ptr pUserContext = GetUserContext( hiRemote );
    if ( pUserContext )
    {
        strName = pUserContext->GetUserName();
    }

    return strName;

}

void CUserMgr::RemoveUserContext(const CHostInfo& hiRemote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    //get the user_context by host info & remove the user_context from map_hi_contexts_
    map<CHostInfo, CUserContext_ptr>::iterator it = user_contexts_.find(hiRemote);
    if( it != user_contexts_.end() )
    {
        user_contexts_.erase(it);
    }
}

bool CUserMgr::ON_CuLoginRequest( ITCPSessionSendSink* sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuLoginReq& req, CuLoginResp& resp)
{
    bool ret = false;
    do 
    {
        //检查连接是否已登录
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<CHostInfo, CUserContext_ptr>::iterator it = user_contexts_.find(hiRemote);
            if( it != user_contexts_.end() )
            {
                Warn("from(%s), user_name(%s), the session has already logined!",
                    hiRemote.GetNodeString().c_str(), req.user_name.c_str());
                break;
            }
        }

        //创建UserContext
        CUserContext_ptr pUserContext = CUserContext_ptr(new CUserContext());
        if( !pUserContext )
        {
            Warn("from(%s), user_name(%s), create user context failed!",
                hiRemote.GetNodeString().c_str(), req.user_name.c_str());
            break;
        }

        if( !pUserContext->ON_CuLoginRequest( sink, hiRemote, msg_seq, req, resp))
        {
            break;
        }

        //增加UserContext
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            user_contexts_.insert( make_pair(hiRemote, pUserContext) );
        }

        ret = true;

    } while (0);

    return ret;
}

bool CUserMgr::ON_CuMediaOpenRequest(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaOpenReq& req)
{
    do
    {
        CUserContext_ptr pUserContext = GetUserContext(hiRemote);
        if(!pUserContext)
        {
            Warn("from(%s), connection cannot found!",
                hiRemote.GetNodeString().c_str() );
            break;
        }

        return pUserContext->ON_CuMediaOpenRequest( sink, hiRemote, msg_seq, req);

    }while(false);

    return false;
}

bool CUserMgr::ON_CuMediaCloseRequest(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaCloseReq& req,  CuMediaCloseResp& resp)
{
    do
    {
        CUserContext_ptr pUserContext = GetUserContext(hiRemote);
        if(!pUserContext)
        {
            Warn("from(%s), connection cannot found!",
                hiRemote.GetNodeString().c_str() );
            break;
        }

        return pUserContext->ON_CuMediaCloseRequest( sink, hiRemote, msg_seq, req, resp );

    }while(false);

    return false;
}

bool CUserMgr::ON_CuStatusReport( ITCPSessionSendSink* sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuStatusReportReq& req, CuStatusReportResp& resp )
{
    do
    {
        CUserContext_ptr pUserContext = GetUserContext(hiRemote);
        if(!pUserContext)
        {
            Warn("from(%s), connection cannot found!",
                hiRemote.GetNodeString().c_str() );
            break;
        }

        return pUserContext->ON_CuStatusReport( sink, hiRemote, msg_seq, req, resp );

    }while(false);

    return false;
}

bool CUserMgr::OnWebUserRequest(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq)
{
    Debug("from(%s), receive web request, url(%s).", hiRemote.GetNodeString().c_str(), pReq->header_detail->url_.c_str() );

    map<string, string> req_params = pReq->header_detail->url_detail_.params_;
    string req_page = pReq->header_detail->url_detail_.page_;

    CWebUserContext_ptr pWebContext = CWebUserContext_ptr(new CWebUserContext(sink, hiRemote, pReq));
    if(!pWebContext)
    {
        return false;
    }

    if(req_page == "/live")
    {
        SDeviceChannel dc;
        dc.device_id_ = req_params["device_id"];
        dc.channel_id_ = (uint16)boost::lexical_cast<int>(req_params["channel_id"]);
        dc.stream_id_ = (uint8)boost::lexical_cast<int>(req_params["stream_id"]);

        CWebUserContext_ptr pWebCtx = CWebUserContext_ptr( new CWebUserContext_Live(sink, hiRemote, pReq) );
        IMediaSessionSink_ptr pSink = boost::dynamic_pointer_cast<IMediaSessionSink>(pWebCtx);
        CMediaSessionMgr_ptr pMediaMgr = GetService()->GetMediaSessionMgr();
        if( pSink && pMediaMgr->OnUserMediaOpenReq(dc, en_session_type_live, hiRemote, pSink) )
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            webcontexts_.insert( make_pair( hiRemote, pWebCtx ) );
        }
        else
        {
            Error("from(%s), media open failed, sink(%p), url(%s).", 
                hiRemote.GetNodeString().c_str(), 
                pSink,
                pReq->header_detail->url_.c_str() );
            return false;
        }
    }
    else if(req_page == "/snap")
    {
        string device_id = req_params["device_id"];
        uint16 channel_id = (uint16)boost::lexical_cast<int>(req_params["channel_id"]);

        CDeviceMgr_ptr pDeviceMgr = CServerLogical::GetLogical()->GetDeviceMgr();
        CDeviceContext_ptr pDeviceCtx = pDeviceMgr->GetDeviceContext(device_id);
        if ( !pDeviceCtx )
        {
            Error("device(%s), device is not online!", device_id.c_str());
            return false;
        }

        CWebUserContext_ptr pWebCtx = CWebUserContext_ptr( new CWebUserContext_Snap(sink, hiRemote, pReq) );
        IDeviceSnapSink_ptr pSink = boost::dynamic_pointer_cast<IDeviceSnapSink>(pWebCtx);
        if( pSink && pDeviceCtx->Screenshot( device_id, channel_id, pSink) )
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            webcontexts_.insert( make_pair( hiRemote, pWebCtx ) );
        }
        else
        {
            Error("from(%s), snap failed, sink(%p), url(%s).", 
                hiRemote.GetNodeString().c_str(), 
                pSink,
                pReq->header_detail->url_.c_str() );
            return false;
        }
    }
    else
    {
        return false;
    }

    return true;
}
