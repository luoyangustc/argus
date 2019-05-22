#include "logging_posix.h"
#include "variant.h"
#include "WebUserContext.h"
#include "ServerLogical.h"
#include "MediaSessionMgr.h"

CWebUserContext::CWebUserContext( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq )
{
    running_ = true;

    last_active_tick_ = get_current_tick();

    send_sink_ = sink;
    hi_remote_ = hiRemote;
    http_req_ = pReq;

    timeout_ms_ = 5*1000;

    Debug("from(%s), construct CWebUserContext object, url(%s).", 
        hiRemote.GetNodeString().c_str(),
        http_req_->header_detail->url_.c_str() );
}

CWebUserContext::~CWebUserContext()
{
}

bool CWebUserContext::IsAlive()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( !running_ )
    {
        return false;
    }

    return true;
}

void CWebUserContext::Update()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( !running_ )
    {
        return;
    }
    
    if( get_current_tick() - last_active_tick_ > timeout_ms_ )
    {
        running_ = false;

        HandleTimeout();

        Error("from(%s), handle http request timeout, url(%s)!", 
            hi_remote_.GetNodeString().c_str(), 
            http_req_->header_detail->url_.c_str() );
    }
}

void CWebUserContext::OnTcpClose(const CHostInfo& hiRemote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    running_ = false;
}

bool CWebUserContext::SendResponse( SHttpResponsePara_ptr pResp )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( !running_ )
    {
        return false;
    }

    send_sink_->SendHttpResp(pResp);

    running_ = false; //After sending the response message, close the session!

    return true;
}

bool CWebUserContext::SendResponse(const string& resp_code, const string& content_type, const uint8* content, size_t content_len)
{
    SHttpResponsePara_ptr pResp(new SHttpResponsePara());
    if( content_len )
    {
        pResp->pContent = boost::shared_array<uint8>( new uint8[content_len+1] );
        memcpy( pResp->pContent.get(), content, content_len );

        pResp->content_len = content_len;
        pResp->content_type = content_type;
    }
    pResp->ret_code = resp_code;

    return SendResponse(pResp);
}

bool CWebUserContext::HandleTimeout()
{
    Variant data;

    Variant resp;
    resp["code"] = -1;
    resp["msg"] = "handle req timeout";
    resp["data"] = data;

    string resp_json;
    if( !resp.SerializeToJSON(resp_json) )
    {
        return false;
    }

    return SendResponse("200 OK", "application/json", (uint8*)resp_json.c_str(), resp_json.length() );
}

CWebUserContext_Live::CWebUserContext_Live( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq )
    : CWebUserContext( sink, hiRemote, pReq )
{
    map<string, string> req_params = pReq->header_detail->url_detail_.params_;

    dc_.device_id_ = req_params["device_id"];
    dc_.channel_id_ = (uint16)boost::lexical_cast<int>(req_params["channel_id"]);
    dc_.stream_id_ = (uint8)boost::lexical_cast<int>(req_params["stream_id"]);
    req_type_ = req_params["type"];

    timeout_ms_ = 5*1000;
}

CWebUserContext_Live::~CWebUserContext_Live()
{
    
}

//bool CWebUserContext::OnDeviceMediaOpenAck(const DeviceMediaOpenResp& resp)
void CWebUserContext_Live::OnMediaOpenAck(
                                int err_code, 
                                const SDeviceChannel& dc, 
                                SessionType session_type, 
                                const string& session_id, 
                                const CHostInfo& hi_stream_serv )
{
    Debug("receive media open ack, dc(%s), session_type(%d), err_code(%d).", 
        dc.GetString().c_str(), 
        (int)session_type, 
        err_code );
    
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if ( !running_ )
        {
            return;
        }
    }

    int code = 0;
    string msg = "success";
    Variant data;

    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if( err_code != 0 )
        {
            code = err_code;
            msg = "open media fail";
            break;
        }

        CHostInfo hiStreamHttp( hi_stream_serv );
        hiStreamHttp.Port += 10;
        if(http_req_->header_detail->url_detail_.page_ == "/live")
        {
            (void)SendWebLiveReqToStream(hiStreamHttp);
        }

        return;

    } while (0);
    
    Variant resp;
    resp["code"] = code;
    resp["msg"] = msg;
    resp["data"] = data;

    string resp_json;
    if( !resp.SerializeToJSON(resp_json) )
    {
        return;
    }

    SendResponse("200 OK", "application/json", (uint8*)resp_json.c_str(), resp_json.length() );
}

bool CWebUserContext_Live::SendWebLiveReqToStream(const CHostInfo& hiRemote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if ( !running_ )
    {
        return false;
    }
    last_active_tick_ = get_current_tick();

    char url[512];
    snprintf( url, sizeof(url), "http://%s/live?device_id=%s&channel_id=%d&stream_id=%d&type=%s",
              hiRemote.GetNodeString().c_str(),
              dc_.device_id_.c_str(),
              (int)dc_.channel_id_,
              (int)dc_.stream_id_,
              req_type_.c_str() );

    uint32_t request_id;
    int ret = WebRequest::instance().SubmitHttpRequest(url, &request_id, OnWebLiveRespFromStream, this, NULL, 0, 2, 3);
    if (ret != 0)
    {
        Error("send live request msg to stream failed, ret(%s). url(%s)", ret, url);
        return false;
    }

    Info("send live request msg to stream success, request_id(%u), url(%s)", request_id, url);
    return true;
}

int CWebUserContext_Live::OnWebLiveRespFromStream(uint32_t request_id, void* user_data, int err_code, int http_code, const char* http_resp)
{
    Debug( "recv http live response,request_id(%u), err_code(%d), http_code(%d), http_resp(%s)", 
        request_id, err_code, http_code, http_resp?http_resp:"");

    /*if ( err_code == 0 && ( http_code>=200 && http_code<300 ) )
    {
        
    }*/

    string resp_code = boost::lexical_cast<string>(http_code);
    CWebUserContext* pContext = (CWebUserContext*)user_data; 
    pContext->SendResponse(resp_code, "application/json", (uint8*)http_resp, strlen(http_resp) );

    return 0;
}

CWebUserContext_Snap::CWebUserContext_Snap( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq )
    : CWebUserContext( sink, hiRemote, pReq )
{
    map<string, string> req_params = pReq->header_detail->url_detail_.params_;

    device_id_ = req_params["device_id"];
    channel_id_ = (uint16)boost::lexical_cast<int>(req_params["channel_id"]);

    if( req_params.find("preview") != req_params.end() && req_params["preview"] == "1" )
    {
        is_preview_ = true;
    }
    else
    {
        is_preview_ = false;
    }

    timeout_ms_ = 25*1000;
}

CWebUserContext_Snap::~CWebUserContext_Snap()
{

}


void CWebUserContext_Snap::OnDeviceSnapAck(int err_code, const string& err_msg, const string& pic_url)
{
    Debug("receive device snap ack, device_id(%s), channel_id(%d), err_code(%d), pic_url(%s).", 
        device_id_.c_str(), 
        (int)channel_id_, 
        err_code,
        pic_url.c_str() );

    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if ( !running_ )
        {
            return;
        }
    }

    int code = 0;
    string msg = "success";
    Variant data;

    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if( err_code != 0 )
        {
            code = err_code;
            msg = err_msg;
            break;
        }

        if( !is_preview_ )
        {
            data["pic_url"] = pic_url;
            break;
        }
        
        SHttpResponsePara_ptr pResp(new SHttpResponsePara());
        pResp->location = pic_url;
        pResp->ret_code = "302 Object Moved";
        (void)SendResponse(pResp);
        return;
    } while (0);

    Variant resp;
    resp["code"] = code;
    resp["msg"] = msg;
    resp["data"] = data;

    string resp_json;
    if( !resp.SerializeToJSON(resp_json) )
    {
        return;
    }

    SendResponse("200 OK", "application/json", (uint8*)resp_json.c_str(), resp_json.length() );
}
