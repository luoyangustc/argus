
#include "variant.h"
#include "ParamParser.h"
#include "http_header_util.h"
#include "GetTickCount.h"
#include "CriticalSectionMgr.h"
#include "ServerLogical.h"

using namespace std;

enum en_resp_code
{
    resp_code_success         = 0,
    resp_code_server_err      = 10001,
    resp_code_req_nosupport   = 10002,
    resp_code_req_param_err   = 10003,
    resp_code_device_offline  = 10004,
    resp_code_channel_offline = 10005,
};

int32 CServerLogical::OnHttpClientRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    do
    {
        if( !pReq->header_detail->is_request )
        {
            Error("from(%s), receive illegal messages!\n", hiRemote.GetNodeString().c_str() );
            break;
        }

        Debug("from(%s), request url=%s\n", hiRemote.GetNodeString().c_str(), pReq->header_detail->url_.c_str());

        if(pReq->header_detail->url_== "/")
        {
            return OnHttpRootRequest(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/favicon.ico" )
        {
            return OnHttpFaviconRequest(sink, hiRemote, pReq, pResp);
        }
        else if(pReq->header_detail->url_detail_.page_ == "/clould_storage")
        {
            return OnHttpCloudStorage(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/live" )
        {
            return OnHttpLiveRequest(sink, hiRemote, pReq, pResp);
        }
        else if (pReq->header_detail->url_detail_.page_ == "/dumpinfo" )
        {
            return OnHttpDumpInfoRequest(sink, hiRemote, pReq, pResp);
        }
        else
        {
            pResp->ret_code = "404 Not Found";
            pResp->keep_alive = false;
        }
        return 0;
    } while (false);
    return -1;
}

int CServerLogical::OnHttpRootRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    static string content_text = "<html><title>result</title><body>Qiniu Stream Server!</body></html>";
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

int CServerLogical::OnHttpCloudStorage(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    //http://ip:port/clould_storage?device_id=x&channel_id=x&rate=x&cmd=start/stop&type=xx
    do
    {
        map<string,string>& params = pReq->header_detail->url_detail_.params_;

        SDeviceChannel dc;
        //get device_id
        map<string,string>::iterator it = params.find("device_id");
        if (it==params.end())
        {
            Error("from(%s), cannot find device id!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        dc.device_id_ = it->second;

        //get channel id
        it = params.find("channel_id");
        if (it==params.end())
        {
            Error("from(%s), cannot find channel id!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        dc.channel_id_ = (uint16)boost::lexical_cast<int>(it->second.c_str());

        //get stream_id
        it = params.find("stream_id");
        if (it==params.end())
        {
            Error("from(%s), cannot find upload rate!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        dc.stream_id_ = (uint8)boost::lexical_cast<int>(it->second.c_str());

        //get command
        it = params.find("cmd");
        if (it==params.end())
        {
            Error("from(%s), cannot find cmd!\n", hiRemote.GetNodeString().c_str() );
            break;
        }

        CMediaSessionBase_ptr pSession = pMediaSessionMgr_->GetMediaSession(MEDIA_SESSION_TYPE_LIVE, dc);
        if(!pSession)
        {
            Error("from(%s), cannot find media session, dc(%s)!\n", hiRemote.GetNodeString().c_str(), dc.GetString().c_str() );
            break;
        }

        string cmd = it->second;
        if(cmd == "start")
        {
            CMediaSessionLive* live_session = reinterpret_cast<CMediaSessionLive*>(pSession.get());
            live_session->SetCloudStorage(true);
        }
        else if(cmd == "stop")
        {
            CMediaSessionLive* live_session = reinterpret_cast<CMediaSessionLive*>(pSession.get());
            live_session->SetCloudStorage(false);
        }
        else
        {
            Error("from(%s), cmd(%s) is incorrect!\n", hiRemote.GetNodeString().c_str(), cmd.c_str() );
            break;
        }
        return 0;
    } while (0);
    return -1;
}

int CServerLogical::OnHttpLiveRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    int code = 0;
    string msg = "success";
    Variant reply;

    do
    {
        //url: http://ip:port/live?device_id=x&channel_id=x&stream_id=x&&type=xx
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
            msg = "cannot find stream_id parma";
            Error("from(%s), cannot find upload rate!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        dc.stream_id_ = (uint8)boost::lexical_cast<int>(it->second.c_str());

        CMediaSessionBase_ptr pSession = pMediaSessionMgr_->GetMediaSession( MEDIA_SESSION_TYPE_LIVE, dc );
        if(!pSession)
        {
            code = resp_code_server_err;
            msg = "get live media session failed";
            Error("from(%s), cannot find media session, dc(%s)!\n", hiRemote.GetNodeString().c_str(), dc.GetString().c_str() );
            break;
        }

        //get command
        it = params.find("type");
        if (it==params.end())
        {
            code = resp_code_req_param_err;
            msg = "cannot find type parma";
            Error("from(%s), cannot find type!\n", hiRemote.GetNodeString().c_str() );
            break;
        }
        string type = it->second;
        if( type == "hflv" || type == "hls" || type == "rtmp" )
        {
            CMediaSessionLive* live_session = reinterpret_cast<CMediaSessionLive*>(pSession.get());
            if ( live_session->SetRtmpLiveAgent(true) < 0 )
            {
                code = resp_code_server_err;
                msg = "open rtmp live agent failed";
                break;
            }

            string rtmp_pub_host = this->pSevrCfg_->GetRtmpHost();
            if( rtmp_pub_host.compare(0, 9, "127.0.0.1") == 0 )
            {
                string str_loc_ip = *this->pSevrCfg_->GetListenIpList().begin();
                rtmp_pub_host.replace(0, 9, str_loc_ip);
            }

            char szPlayUrl[1024];
            if ( type == "hflv" )
            {
                int len = snprintf(szPlayUrl, sizeof(szPlayUrl)-1, 
                            "http://%s:%u/live?port=%u&app=live&stream=%s",
                            rtmp_pub_host.c_str(),
                            pSevrCfg_->GetRtmpHttpPort(),
                            pSevrCfg_->GetRtmpPort(),
                            dc.GetString().c_str() );
                if(len < 0)
                {
                    code = resp_code_server_err;
                    msg = "generate hflv url failed";
                    break;
                }
                szPlayUrl[len] = '\0';
            }
            else if ( type == "hls" )
            {
                int len = snprintf(szPlayUrl, sizeof(szPlayUrl)-1,
                    "http://%s:%u/%s/%s.m3u8",
                    rtmp_pub_host.c_str(),
                    pSevrCfg_->GetRtmpHttpPort(),
                    pSevrCfg_->GetRtmpHlsPath().c_str(),
                    dc.GetString().c_str() );
                if(len < 0)
                {
                    code = resp_code_server_err;
                    msg = "generate hls url failed";
                    break;
                }
                szPlayUrl[len] = '\0';
            }
            else if ( type == "rtmp" )
            {
                string strParams;
                const map<string, string>& play_params = this->pSevrCfg_->GetRtmpPlayParams();
                if( !play_params.empty() )
                {
                    bool is_fist = true;
                    map<string, string>::const_iterator it = play_params.begin();
                    for ( ; it != play_params.end(); ++it )
                    {
                        if ( is_fist )
                        {
                            is_fist = false;
                        }
                        else
                        {
                            strParams += "&";
                        }

                        strParams += it->first + "=" + it->second;
                    }
                }

                string strPath = pSevrCfg_->GetRtmpPath();
                if(!strPath.empty())
                {
                    strPath += "/";
                }

                int len = snprintf(szPlayUrl, sizeof(szPlayUrl)-1,
                    "rtmp://%s:%u/%slive/%s",
                    rtmp_pub_host.c_str(),
                    pSevrCfg_->GetRtmpPort(),
                    strPath.c_str(),
                    dc.GetString().c_str() );
                if(len < 0)
                {
                    code = resp_code_server_err;
                    msg = "generate rtmp url failed";
                    break;
                }
                szPlayUrl[len] = '\0';
            }

            Variant data;
            data["play_url"] = szPlayUrl;
            reply["data"] = data;

        }
        else if(type == "close")
        {
            CMediaSessionLive* live_session = reinterpret_cast<CMediaSessionLive*>(pSession.get());
            live_session->SetRtmpLiveAgent(false);
        }
        else
        {
            code = -1;
            msg = "not support type:";
            msg += type;
            Error("from(%s), type(%s) is incorrect!\n", hiRemote.GetNodeString().c_str(), type.c_str() );
            break;
        }
    } while (0);

    reply["code"] = code;
    reply["message"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    pResp->pContent = boost::shared_array<uint8>(new uint8[content.length()+1]);
    memcpy(pResp->pContent.get(), content.c_str(), content.length());
    pResp->content_len = content.length();
    pResp->ret_code = "200 OK";
    pResp->content_type = "application/json";

    Debug("from(%s), response msg(%s).\n", hiRemote.GetNodeString().c_str(), content.c_str() );

    return 0;
}


int CServerLogical::OnHttpDumpInfoRequest(ITCPSessionSendSink*sink, CHostInfo& hiRemote, SHttpRequestPara_ptr pReq, SHttpResponsePara_ptr pResp)
{
    if( !pMediaSessionMgr_ )
    {
        return -1;
    }

    Variant reply;
    map<string,string>& params = pReq->header_detail->url_detail_.params_;
    map<string,string>::iterator it = params.find("device_id");
    if ( params.find("device_id") != params.end() )
    {
        if( params.find("channel_id") != params.end() &&
            params.find("stream_id") != params.end() )
        {
            SDeviceChannel dc;
            dc.device_id_ = params["device_id"];
            dc.channel_id_ = atoi(params["channel_id"].c_str());
            dc.stream_id_ = atoi(params["stream_id"].c_str());
            pMediaSessionMgr_->DumpInfo(dc, reply);
        }
        else
        {
            pMediaSessionMgr_->DumpInfo(params["device_id"], reply);
        }
    }
    else
    {
        pMediaSessionMgr_->DumpInfo(reply);
    }

    std::string content;
    reply.SerializeToJSON(content);

    pResp->pContent = boost::shared_array<uint8>(new uint8[content.length()+1]);
    memcpy(pResp->pContent.get(), content.c_str(), content.length());
    pResp->content_len = content.length();
    pResp->ret_code = "200 OK";
    pResp->content_type = "application/json";

    //Debug("from(%s), response msg(%s).\n", hiRemote.GetNodeString().c_str(), content.c_str() );
    return 0;
}