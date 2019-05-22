#include <boost/lexical_cast.hpp>
#include "LFile.h"
#include "logging_posix.h"
#include "variant.h"
#include "ParamParser.h"
#include "http_header_util.h"
#include "GetTickCount.h"
#include "CriticalSectionMgr.h"
#include "ServerLogical.h"
#include "MediaSession.h"

enum en_resp_code
{
    resp_code_success         = 0,
    resp_code_server_err      = 10001,
    resp_code_req_nosupport   = 10002,
    resp_code_req_param_err   = 10003,
    resp_code_device_offline  = 10004,
    resp_code_channel_offline = 10005,
};

int CServerLogical::OnHttpClientRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote,SHttpRequestPara_ptr pReq,SHttpResponsePara_ptr pResp)
{
    do
    {
        if( ! pReq->header_detail->is_request )
        {
            break;
        }

        if(pReq->header_detail->url_== "/")
        {
            return OnHttpRootRequest(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/favicon.ico" )
        {
            return OnHttpFaviconRequest(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/snap")
        {
            return OnHttpSnapRequest(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/ptz_ctrl")
        {
            return OnHttpPtzctrlRequest(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/clould_storage")
        {
            return OnHttpCloudStorage(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/live" )
        {
            return OnHttpLiveRequest(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/device_mgr_update" )
        {
            return OnHttpDeviceMgrUpdate(sink, hiRemote, pReq, pResp);
        }
        else
        {
            pResp->ret_code = "404 Not Found";
            pResp->keep_alive = false;
            return 0;
        }
    }
    while(false);

    return -1;
}

int CServerLogical::OnHttpRootRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    static string content_text = "<html><title>result</title><body>Qiniu Session Server!</body></html>";
    pResp->pContent = boost::shared_array<uint8>(new uint8[content_text.length()+1]);
    memcpy(pResp->pContent.get(), content_text.c_str(), content_text.length()+1);
    pResp->content_len = content_text.length();
    pResp->ret_code = "200 OK";
    pResp->content_type = "text/html";
    return 0;
}

int CServerLogical::OnHttpFaviconRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    static boost::shared_array<BYTE> favicon_content;
    static unsigned int favicon_content_len = 0;
    static CCriticalSectionMgr citical_section;
    {
        CCriticalSection lock(&citical_section);
        if (!favicon_content)
        {
            string favicon_file = CLFile::GetModuleDirectory() + "/favicon.ico";;
            FILE * fp = fopen(favicon_file.c_str(),"rb+");
            if (fp)
            {
                fseek(fp,0,SEEK_SET);
                fseek(fp,0,SEEK_END);
                long longBytes=ftell(fp);// longBytes就是文件的长度
                favicon_content =   boost::shared_array<BYTE>(new BYTE[longBytes]);
                if (favicon_content)
                {
                    favicon_content_len = longBytes;
                    fseek(fp,0,SEEK_SET);
                    fread(favicon_content.get(),longBytes,1,fp);
                }																							   						
                fclose(fp);
            }

        }
    }		
    if (favicon_content)
    {
        pResp->pContent = boost::shared_array<uint8>(new uint8[favicon_content_len+1]);
        memcpy(pResp->pContent.get(),favicon_content.get(),favicon_content_len+1);
        pResp->content_len = favicon_content_len;
        pResp->ret_code = "200 OK";
        pResp->keep_alive = true;
        pResp->content_type = "image/x-icon";//application/x-ico
    }

    return 0;
}

int CServerLogical::OnHttpSnapRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    int code = 0;
    string msg = "success";
    do 
    {
        map<string,string>& params = pReq->header_detail->url_detail_.params_;

        string device_id;
        int channel_index;
        if ( params.find("device_id")==params.end() ||
            params.find("channel_id")==params.end() )
        {
            code = resp_code_req_param_err;
            msg = "params incorrect";
            break;
        }
        device_id = params["device_id"];

        try
        {
            channel_index = boost::lexical_cast<int>(params["channel_id"]);   
        }
        catch(...)
        {
            code = resp_code_req_param_err;
            msg = "channel_id is incorrect";
            break;
        }

        string& pic_save_path = GetService()->GetServCfg()->GetSnapPicSavePath();
        string& pic_url =  GetService()->GetServCfg()->GetSnapPicUrl();
        if( pic_save_path.empty() || pic_url.empty() )
        {
            code = resp_code_server_err;
            msg = "snap config error!";
            break;
        }

        if( !pUserMgr_->OnWebUserRequest(sink, hiRemote, pReq) )
        {
            code = resp_code_server_err;
            msg = "handle snap request failed";
            Error("from(%s), handle snap request failed, did(%s), channel_id(%d)!\n", 
                hiRemote.GetNodeString().c_str(), device_id.c_str(), channel_index );
            break;
        }

        //Debug( "from(%s), device_id(%s), channel_index(%d).", 
        //    hiRemote.GetNodeString().c_str(), 
        //    device_id.c_str(), 
        //    channel_index );
        return 0;
    } while (0);
    
    Variant reply;
    reply["code"] = code;
    reply["msg"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    pResp->pContent = boost::shared_array<uint8>(new uint8[content.length()+1]);
    memcpy(pResp->pContent.get(), content.c_str(), content.length());
    pResp->content_len = content.length();
    pResp->ret_code = "200 OK";
    pResp->content_type = "application/json";

    return 0;
}

int CServerLogical::OnHttpPtzctrlRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    int ret = -1;
    int code = 0;
    string msg = "success";
    do 
    {
        map<string,string>& params = pReq->header_detail->url_detail_.params_;

        string device_id;
        int channel_index, ptz_cmd;
        if ( params.find("device_id") == params.end() ||
             params.find("channel_id") == params.end() ||
             params.find("cmd_type") == params.end() )
        {
            code = resp_code_req_param_err;
            msg = "params incorrect";
            break;
        }
        device_id = params["device_id"];
        try
        {
            channel_index = boost::lexical_cast<int>(params["channel_id"]); 
            ptz_cmd = boost::lexical_cast<int>(params["cmd_type"]);
        }
        catch(...)
        {
            code = resp_code_req_param_err;
            msg = "channel_id or cmd_type is incorrect";
            break;
        }

        CDeviceContext_ptr pDevCtx = this->GetDeviceContext(device_id);
        if ( !pDevCtx )
        {
            code = resp_code_device_offline;
            msg = "device is offline";
            break;
        }

        if ( !pDevCtx->PtzCtrl(device_id, channel_index, ptz_cmd) )
        {
            code = resp_code_server_err;
            msg = "handle screen shot failed";
            break;
        }

        Debug( "from(%s), device_id(%s), channel_index(%d), prz_cmd(%d).", 
            hiRemote.GetNodeString().c_str(), 
            device_id.c_str(), 
            channel_index, 
            ptz_cmd );

        ret = 0;
    } while (0);

    Variant reply;
    reply["code"] = code;
    reply["msg"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    pResp->pContent = boost::shared_array<uint8>(new uint8[content.length()+1]);
    memcpy(pResp->pContent.get(), content.c_str(), content.length());
    pResp->content_len = content.length();
    pResp->ret_code = "200 OK";
    pResp->content_type = "application/json";

    return 0;
}

int CServerLogical::OnHttpCloudStorage(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    //http://ip:port/clould_storage?device_id=x&channel_id=x&rate=x&cmd=start/stop&type=xx
    do
    {
        map<string,string>& params = pReq->header_detail->url_detail_.params_;
        
        //TODO:

        return 0;
    } while (0);
    return -1;
}

int CServerLogical::OnHttpLiveRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    int code = 0, ret = -1;
    string msg = "success";
    Variant reply;

    do
    {
        //url: http://ip:port/live?device_id=x&channel_id=x&rate=x&&type=xx
        //param: type[hflv, hls, rtmp, close]

        map<string,string>& params = pReq->header_detail->url_detail_.params_;
        SDeviceChannel dc;
        //get device_id
        map<string,string>::iterator it = params.find("device_id");
        if (it==params.end())
        {
            code = resp_code_req_param_err;
            msg = "cannot find device_id parma";
            Error("from(%s), cannot find device id!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        dc.device_id_ = it->second;

        //get channel id
        it = params.find("channel_id");
        if (it==params.end())
        {
            code = resp_code_req_param_err;
            msg = "cannot find channel_id parma";
            Error("from(%s), cannot find channel id!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        dc.channel_id_ = (uint16)boost::lexical_cast<int>(it->second.c_str());

        //get stream_id
        it = params.find("stream_id");
        if (it==params.end())
        {
            code = resp_code_req_param_err;
            msg = "cannot find rate parma";
            Error("from(%s), cannot find upload rate!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        dc.stream_id_ = (uint8)boost::lexical_cast<int>(it->second.c_str());

        //get command
        it = params.find("type");
        if (it==params.end())
        {
            code = resp_code_req_param_err;
            msg = "cannot find type parma, dc(%s)";
            Error("from(%s), cannot find type!\n", hiRemote.GetNodeString().c_str(), dc.GetString().c_str() );
            break;
        }

        CDeviceMgr_ptr pDeviceMgr = CServerLogical::GetLogical()->GetDeviceMgr();
        CDeviceContext_ptr pDeviceCtx = pDeviceMgr->GetDeviceContext(dc.device_id_);
        if ( !pDeviceCtx )
        {
            code = resp_code_device_offline;
            msg = "device offline";
            Error("from(%s), device is not online, dc(%s)!", hiRemote.GetNodeString().c_str(),dc.GetString().c_str() );
            break;
        }

        protocol::DevChannelInfo channel_info;
        if( !pDeviceCtx->GetChannel(dc.channel_id_, channel_info) )
        {
            code = resp_code_req_param_err;
            msg = "channel_id incorrect";
            Error("from(%s), get channel failed, dc(%s), max_channel_id(%d).", 
                hiRemote.GetNodeString().c_str(), dc.GetString().c_str(), (int)pDeviceCtx->GetChannelNum());
            break;
        }

        if( channel_info.channel_status == protocol::CHANNEL_STS_OFFLINE )
        {
            code = resp_code_device_offline;
            msg = "channel offline";
            Error("from(%s), channel is not online, dc(%s)!", hiRemote.GetNodeString().c_str(), dc.GetString().c_str());
            break;
        }

        if( channel_info.stream_num == 0 || 
            dc.stream_id_ >= channel_info.stream_list.size() )
        {
            code = resp_code_req_param_err;
            msg = "stream_id incorrect";
            Error("from(%s), check stream_id faied, dc(%s), stream_num(%d).", 
                hiRemote.GetNodeString().c_str(), dc.GetString().c_str(), (int)channel_info.stream_num );
            break;
        }

        if( !pUserMgr_->OnWebUserRequest(sink, hiRemote, pReq) )
        {
            code = resp_code_server_err;
            msg = "handle live request failed";
            Error("from(%s), handle live request failed, dc(%s)!\n", hiRemote.GetNodeString().c_str(), dc.GetString().c_str() );
            break;
        }

        return 0;

    } while (0);

    reply["code"] = code;
    reply["msg"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    pResp->pContent = boost::shared_array<uint8>(new uint8[content.length()+1]);
    memcpy(pResp->pContent.get(), content.c_str(), content.length());
    pResp->content_len = content.length();
    pResp->ret_code = "200 OK";
    pResp->content_type = "application/json";

    return 0;
}

int CServerLogical::OnHttpDeviceMgrUpdate(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    int ret = -1;
    int code = 0;
    string msg = "success";
    do 
    {
        map<string,string>& params = pReq->header_detail->url_detail_.params_;

        string device_id;
        int channel_index, mgr_type;
        if ( params.find("device_id") == params.end() ||
            params.find("channel_id") == params.end() ||
            params.find("mgr_type") == params.end() )
        {
            code = resp_code_req_param_err;
            msg = "params incorrect";
            break;
        }
        device_id = params["device_id"];
        try
        {
            channel_index = boost::lexical_cast<int>(params["channel_id"]); 
            mgr_type = boost::lexical_cast<int>(params["mgr_type"]);
        }
        catch(...)
        {
            code = resp_code_req_param_err;
            msg = "channel_id or cmd_type is incorrect";
            break;
        }

        CDeviceContext_ptr pDevCtx = this->GetDeviceContext(device_id);
        if ( !pDevCtx )
        {
            code = resp_code_device_offline;
            msg = "device is offline";
            break;
        }

        if ( !pDevCtx->MgrUpdate(device_id, channel_index, mgr_type) )
        {
            code = resp_code_server_err;
            msg = "handle screen shot failed";
            break;
        }

        Debug( "from(%s), device_id(%s), channel_index(%d), mgr_type(%d).", 
            hiRemote.GetNodeString().c_str(), 
            device_id.c_str(), 
            channel_index, 
            mgr_type );

        ret = 0;
    } while (0);

    Variant reply;
    reply["code"] = code;
    reply["msg"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    pResp->pContent = boost::shared_array<uint8>(new uint8[content.length()+1]);
    memcpy(pResp->pContent.get(), content.c_str(), content.length());
    pResp->content_len = content.length();
    pResp->ret_code = "200 OK";
    pResp->content_type = "application/json";

    return 0;
}
