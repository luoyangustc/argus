#include "StatusServer.h"
#include "HttpRequestHandler.h"
#include <vector>
#include <boost/algorithm/string.hpp>
#include "base/include/http_header_util.h"
#include "base/include/HostInfo.h"
#include "base/include/logging_posix.h"
#include "base/include/DeviceChannel.h"
#include "base/include/variant.h"
#include "protocol/include/protocol_status.h"
#include "util_module/include/util_module.h"

HttpRequestHandler::HttpRequestHandler(const SessionMgrPtr& session_mgr)
    : session_mgr_(session_mgr)
{
}

HttpRequestHandler:: ~HttpRequestHandler()
{
}

int HttpRequestHandler::OnHttpRequest(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    const std::string root_directory            = "/";
    const std::string gen_device_id             = "/gen_device_id";
    const std::string get_access_serv           = "/get_access_server";
    const std::string get_access_token          = "/get_access_token";
    const std::string get_session_serv          = "/get_session_server";
    const std::string streamrrs                 = "/streamrrs";
    const std::string stream_server_info        = "/stream_server_info.json";
    const std::string session_server_info       = "/session_server_info.json";
    const std::string query_device_status       = "/query_device_status";
    const std::string req_live                  = "/live";
    const std::string device_control            = "/device_control";
    const std::string device_snap               = "/snap";

    std::string page = req->header_detail->url_detail_.page_;
    HttpParameterMap params = req->header_detail->url_detail_.params_;

    if (page == root_directory)
    {
        return OnRoot(peer_addr, req, resp);
    }
    else if (page == gen_device_id)
    {
        return OnGenerateDeviceId(peer_addr, req, resp);
    }
    else if (page == get_session_serv)
    {
        return OnGetSessionServer(peer_addr, req, resp);
    }
    else if (page == get_access_serv)
    {
        return OnGetAccessServer(peer_addr, req, resp);
    }
    else if (page == get_access_token)
    {
        return OnGetAccessToken(peer_addr, req, resp);
    }
    else if (page == streamrrs)
    {
        return OnStreamRRS(peer_addr, req, resp);
    }
    else if (page == stream_server_info)
    {
        return OnQueryStreamServerInfo(peer_addr, req, resp);
    }
    else if (page == session_server_info)
    {
        return OnQuerySessionServerInfo(peer_addr, req, resp);  
    }
    else if (page == query_device_status)
    {				
        return OnQueryDeviceStatus(peer_addr, req, resp);
    }
    else if (page == req_live)
    {
        return OnReqLive(peer_addr, req, resp);
    }
    else if (page == device_control)
    {
        return OnDeviceControl(peer_addr, req, resp);
    }
    else if (page == device_snap)
    {
        return OnDeviceSnap(peer_addr, req, resp);
    }
    else
    {
        return HttpNotFound(resp);
    }
}

int HttpRequestHandler::OnRoot(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    const std::string content = 
        "<html><title>result</title><body>QN Status server!</body></html>";

    MakeHttpResponse(content, resp);
    resp->content_type = "text/html"; 
    return 0;
}

int HttpRequestHandler::OnGenerateDeviceId(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    //url-> http://ip:port/gen_device_id?oem_id=Zz&sn=123aa123

    Variant reply;

    do
    {
        HttpParameterMap& params = req->header_detail->url_detail_.params_;
        HttpParameterIterator it= params.find("oem_id");
        if (it == params.end())
        {
            reply["code"] = -1;
            reply["message"] = "oem_id missing";
            break;
        }
        // step1 : copy oem_id
        char original_did[24] = "";  // original_did's length is 18 bytes  
        const int oem_id_length = 2;  // oem_id's length is 2 bytes  
        if (oem_id_length == it->second.length())  
        {
            memcpy(original_did, it->second.c_str(), oem_id_length);
        }
        else
        {
            reply["code"] = -1;
            reply["message"] = "oem_id length != 2 bytes";
            break;
        }

        // step2 : copy sn  
        it = params.find("sn");
        if (it == params.end())
        {
            reply["code"] = -1;
            reply["message"] = "sn missing";
            break;    
        }

        const int sn_max_length = 16;  // sn's max length is 16bytes
        if (it->second.length() <= sn_max_length)
        {
            if (it->second.length() == sn_max_length)
            {
                memcpy(original_did + oem_id_length, it->second.c_str(), sn_max_length);        
            }
            else
            {
                const int zero_num = sn_max_length - it->second.length();
                for (int index = 0; index < zero_num; ++index)
                {
                    // If the number is not enough, then fill character 'F'
                    original_did[oem_id_length + index] = '0';  // FIXME : F
                }
                memcpy(original_did + oem_id_length + zero_num, it->second.c_str(), it->second.length());
            }  
        }
        else
        {
            reply["code"] = -1;
            reply["message"] = "sn length > 16 bytes";
            break;
        }

        char generate_did[32] = "";
        unsigned int length = sizeof generate_did;
        if ( um_generate_device_id(original_did, generate_did, &length) < 0 )
        {
            reply["code"] = -1;
            reply["message"] = "Server Internal Error";
        }
        else
        {
            Variant data;
            data["device_id"] = generate_did;

            reply["code"] = 0;
            reply["message"] = "Success";
            reply["data"] = data;
        }

    }while(0);

    std::string content;
    reply.SerializeToJSON(content);
    MakeHttpResponse(content, resp);
    return 0;
}

int HttpRequestHandler::OnGetAccessToken(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    // url --> http://%s:%d/cc?device_id=%s

    Variant reply;
    {
        Variant data;
        data["access_token"] = "1fb72cd23620be50f8460fd4f15664da9dee63c2";

        reply["code"] = 0;
        reply["message"] = "Success";
        reply["data"] = data;
    }

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,resp);
    return 0;
}

int HttpRequestHandler::OnGetAccessServer(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    // url http://%s:%d/get_access_server?device_id=%s

    Variant reply;

    do 
    {
        SessionContextPtr pSession = session_mgr_->SelectBestSessionContext(peer_addr, protocol::EP_SMS);
        if( !pSession )
        {
            reply["code"]= -1;
            reply["message"] = "cannot get access server";
            break;
        }

        Variant data;
        data["access_server_ip"] = pSession->listen_ip_list_[0];
        data["access_server_port"] = pSession->serv_port_;

        reply["code"] = 0;
        reply["message"] = "Success";
        reply["data"] = data;
    } while (0);

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,resp);
    return 0; 
}

int HttpRequestHandler::OnGetSessionServer(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    // url http://%s:%d/get_session_server?device_id=%s
    // url http://%s:%d/get_session_server
    Variant reply;

    do 
    {
        SessionContextPtr pSession = session_mgr_->SelectBestSessionContext(peer_addr, protocol::EP_SMS);
        if( !pSession )
        {
            reply["code"]= -1;
            reply["message"] = "cannot get access server";
            break;
        }

        Variant data;
        data["ip"] = pSession->listen_ip_list_[0];
        data["port"] = pSession->serv_port_;
        data["token"] = "";

        reply["code"] = 0;
        reply["msg"] = "Success";
        reply["data"] = data;
    } while (0);

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,resp);
    return 0; 
}

int HttpRequestHandler::OnStreamRRS(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    // url --> http://%s/streamrrs?device_id=%s&client_real_ip=%s&device_ip=%s&business_channel=stream.gslb.qn.com

    Variant reply;

    do 
    {
        SessionContextPtr pSession = session_mgr_->SelectBestSessionContext(peer_addr, protocol::EP_STREAM);
        if( !pSession )
        {
            reply["code"]= -1;
            reply["message"] = "cannot get stream server";
            break;
        }

        Variant server;
        {
            CHostInfo hiStream(pSession->listen_ip_list_[0], pSession->serv_port_);
            Variant stream = hiStream.GetNodeString();
            server.PushToArray(stream);
        }

        Variant data;
        data["server"] = server;

        reply["code"] = 0;
        reply["message"] = "Success";
        reply["data"] = data;
    } while (0);

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,resp);
    return 0;
}

int HttpRequestHandler::OnQueryStreamServerInfo(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    Variant reply;
    vector<SessionContextPtr> stream_ctx;
    session_mgr_->GetSessionContext(protocol::EP_STREAM, stream_ctx);

    reply["session_num"] = stream_ctx.size();
    vector<SessionContextPtr>::iterator it = stream_ctx.begin();
    for( ; it!=stream_ctx.end(); ++it )
    {
        Variant session_info;
        SessionContextPtr pSession = *it;
        session_info["http_port"] = pSession->http_port_;
        session_info["serv_port"] = pSession->serv_port_;

        string listen_ips;
        vector<std::string>::iterator it1 = pSession->listen_ip_list_.begin();
        bool is_first = true;
        for( ; it1 != pSession->listen_ip_list_.end(); ++it1 )
        {
            if(is_first)
            {
                listen_ips += *it1;
                is_first = false;
            }
            else
            {
                listen_ips += ",";
                listen_ips += *it1;
            }
        }
        session_info["listen_ip_list"] = listen_ips;
        session_info["tcp_conn_num"] = pSession->tcp_conn_num_;
        session_info["cpu_use"] = pSession->cpu_use_;
        session_info["mempry_use"] = pSession->mempry_use_;

        reply["sessions"].PushToArray(session_info);
    }
    
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content, resp);
    return 0;
}

int HttpRequestHandler::OnQuerySessionServerInfo(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    Variant reply;
    vector<SessionContextPtr> stream_ctx;
    session_mgr_->GetSessionContext(protocol::EP_SMS, stream_ctx);

    reply["session_num"] = stream_ctx.size();
    vector<SessionContextPtr>::iterator it = stream_ctx.begin();
    for( ; it!=stream_ctx.end(); ++it )
    {
        Variant session_info;
        SessionContextPtr pSession = *it;
        session_info["http_port"] = pSession->http_port_;
        session_info["serv_port"] = pSession->serv_port_;

        string listen_ips;
        vector<std::string>::iterator it1 = pSession->listen_ip_list_.begin();
        bool is_first = true;
        for( ; it1 != pSession->listen_ip_list_.end(); ++it1 )
        {
            if(is_first)
            {
                listen_ips += *it1;
                is_first = false;
            }
            else
            {
                listen_ips += ",";
                listen_ips += *it1;
            }
        }
        session_info["listen_ip_list"] = listen_ips;
        session_info["tcp_conn_num"] = pSession->tcp_conn_num_;
        session_info["cpu_use"] = pSession->cpu_use_;
        session_info["mempry_use"] = pSession->mempry_use_;

        reply["sessions"].PushToArray(session_info);
    }

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content, resp);
    return 0;
}

int HttpRequestHandler::OnQueryDeviceStatus(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    // 1. url --> http://ip:port/query_device_status?device_id=x1,x2,x3
    // 2. url --> http://ip:port/query_device_status?type=x   (x: online/offline/all)

    do
    {
        std::vector<std::string> device_list;

        HttpParameterMap& params = req->header_detail->url_detail_.params_;
        HttpParameterIterator it= params.find("device_id");
        if ( it != params.end() )
        {
            boost::algorithm::split(device_list, it->second, boost::is_any_of("; |,") );
        }
        else
        {
            it = params.find("type");
            if ( it != params.end() )
            {
                if( it->second ==  "online" )
                {
                    GetDeviceMgr()->GetOnlineDeivce(device_list);
                }
                else if( it->second ==  "offline" )
                {
                    GetDeviceMgr()->GetOfflineDeivce(device_list);
                }
                else if( it->second ==  "all" )
                {
                    GetDeviceMgr()->GetOnlineDeivce(device_list);
                    GetDeviceMgr()->GetOfflineDeivce(device_list);
                }
                else
                {
                    break;
                }
            }
        }
        
        Variant reply;
        reply["device_num"] = device_list.size();

        for (int i = 0; i < device_list.size(); ++i)
        {
            const std::string device_id = device_list[i];
            {					
                Variant temp;
                if (! QueryDeviceStatus(device_id,temp))
                {
                    Error("Query device(%s) status error\n",device_id.c_str());
                    temp.Reset();
                }
                reply["devices"].PushToArray(temp);
            }
        }
        // make json response
        std::string content;
        reply.SerializeToJSON(content);

        MakeHttpResponse(content, resp);
        return 0;
    } while (false);

    return NotHandleOk(resp);
}

int HttpRequestHandler::OnReqLive(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    int code = 0, ret = -1;
    string msg = "success";
    Variant reply;

    do
    {
        //url: http://ip:port/live?device_id=x&channel_id=x&stream_id=x&&type=xx
        //param: type[hflv, hls, rtmp, close]

        map<string,string>& params = req->header_detail->url_detail_.params_;
        SDeviceChannel dc;
        //get device_id
        map<string,string>::iterator it = params.find("device_id");
        if (it==params.end())
        {
            code = -1;
            msg = "cannot find device_id parma";
            Error("from(%s), cannot find device id!\n", peer_addr.GetNodeString().c_str() );
            break;
        }
        dc.device_id_ = it->second;

        //get channel id
        it = params.find("channel_id");
        if (it==params.end())
        {
            code = -1;
            msg = "cannot find channel_id parma";
            Error("from(%s), cannot find channel id!\n", peer_addr.GetNodeString().c_str() );
            break;
        }
        dc.channel_id_ = (uint16)boost::lexical_cast<int>(it->second.c_str());

        //get stream_id
        it = params.find("stream_id");
        if (it==params.end())
        {
            code = -1;
            msg = "cannot find rate parma";
            Error("from(%s), cannot find stream id!\n", peer_addr.GetNodeString().c_str() );
            break;
        }
        dc.stream_id_ = (uint8)boost::lexical_cast<int>(it->second.c_str());

        //get command
        it = params.find("type");
        if (it==params.end())
        {
            code = -1;
            msg = "cannot find type parma";
            Error("from(%s), cannot find type!\n", peer_addr.GetNodeString().c_str() );
            break;
        }

        DevicePtr pDevice = GetDeviceMgr()->GetDevice(dc.device_id_);
        if( !pDevice || pDevice->status_ != protocol::SDeviceSessionStatus::enm_dev_status_online )
        {
            code = -1;
            msg = "device is offline";
            Error("from(%s), handle live request failed, dc(%s) offline!\n", peer_addr.GetNodeString().c_str(), dc.GetString().c_str() );
            break;
        }

        CHostInfo hiSessionHttp(pDevice->session_server_addr_);
        hiSessionHttp.Port += 10;

        // make 302 redirect location
        std::string location;
        location.append("http://");
        location.append( hiSessionHttp.GetNodeString() );
        location.append(req->header_detail->url_);

        resp->location = location;
        resp->ret_code = "302 Object Moved";
        return 0;

    } while (0);

    reply["code"] = code;
    reply["message"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content, resp);
    return 0;
}

int HttpRequestHandler::OnDeviceControl(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    int code = 0, ret = -1;
    string msg = "success";
    Variant reply;
    do 
    {
        HttpParameterMap& params = req->header_detail->url_detail_.params_;
        HttpParameterIterator iter= params.find("device_id");
        if (iter == params.end())
        {
            code = -1;
            msg = "cannot find device_id parma";
            Error("from(%s), cannot find device id!\n", peer_addr.GetNodeString().c_str() );
            break;
        }
        const std::string device_id = params["device_id"];
        DevicePtr pDevice = GetDeviceMgr()->GetDevice(device_id);
        if( !pDevice || pDevice->status_ != protocol::SDeviceSessionStatus::enm_dev_status_online )
        {
            code = -1;
            msg = "device is offline";
            Error("from(%s), handle device control request failed, device(%s) offline!\n", 
                peer_addr.GetNodeString().c_str(), device_id.c_str() );
            break;
        }

        CHostInfo hiSessionHttp(pDevice->session_server_addr_);
        hiSessionHttp.Port += 10;

        // make 302 redirect location
        std::string location;
        location.append("http://");
        location.append( hiSessionHttp.GetNodeString() );
        location.append(req->header_detail->url_);

        resp->location = location;
        resp->ret_code = "302 Object Moved";

        return 0;
    } while (false);

    reply["code"] = code;
    reply["message"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content, resp);
    return 0;
}

int HttpRequestHandler::OnDeviceSnap(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp)
{
    // url --> http://ip:port/snap?device_id=ZZ00008C8590CD9D37VO&channel_id=1&preview=1

    int code = 0, ret = -1;
    string msg = "success";
    Variant reply;
    do 
    {
        HttpParameterMap& params = req->header_detail->url_detail_.params_;
        HttpParameterIterator it= params.find("device_id");
        if (it == params.end())
        {
            code = -1;
            msg = "cannot find device_id parma";
            Error("from(%s), cannot find device id!\n", peer_addr.GetNodeString().c_str() );
            break;
        }

        it = params.find("channel_id");
        if (it==params.end())
        {
            code = -1;
            msg = "cannot find channel_id parma";
            Error("from(%s), cannot find channel id!\n", peer_addr.GetNodeString().c_str() );
            break;
        }

        const std::string device_id = params["device_id"];
        DevicePtr pDevice = GetDeviceMgr()->GetDevice(device_id);
        if( !pDevice || pDevice->status_ != protocol::SDeviceSessionStatus::enm_dev_status_online )
        {
            code = -1;
            msg = "device is offline";
            Error("from(%s), handle device snap request failed, device(%s) offline!\n", 
                peer_addr.GetNodeString().c_str(), device_id.c_str() );
            break;
        }

        CHostInfo hiSessionHttp(pDevice->session_server_addr_);
        hiSessionHttp.Port += 10;

        // make 302 redirect location
        std::string location;
        location.append("http://");
        location.append( hiSessionHttp.GetNodeString() );
        location.append(req->header_detail->url_);

        resp->location = location;
        resp->ret_code = "302 Object Moved";

        return 0;
    } while (false);

    reply["code"] = code;
    reply["message"] = msg;

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content, resp);
    return 0;
}

bool HttpRequestHandler::QueryDeviceStatus(const std::string& device_id, Variant& reply)
{
    const char* DeviceTypeStr[protocol::DEV_TYPE_MAX] = {"ipc","nvr","dvr","smartbox"};

    do
    {
        reply["device_id"] = device_id;

        DevicePtr pDevice = GetDeviceMgr()->GetDevice(device_id);
        if(!pDevice)
        {
            reply["status"] = "unknow";
            break;
        }

        if( pDevice->status_ == protocol::SDeviceSessionStatus::enm_dev_status_online )
        {
            reply["status"] = "online";
            reply["timestamp"] = TimestampStr(pDevice->timestamp_);
            if( pDevice->dev_type_ < protocol::DEV_TYPE_MAX )
            {
                reply["dev_type"] = DeviceTypeStr[pDevice->dev_type_];
            }
            else
            {
                reply["dev_type"] = pDevice->dev_type_;
            }

            reply["session_serv_addr"] = pDevice->session_server_addr_.GetNodeString();
            reply["channel_num"] = pDevice->channel_num_;

            string channel_status = "";
            for(int i = 0; i < pDevice->channel_num_; i++ )
            {
                channel_status += "0";
            }

            vector<protocol::DevChannelInfo>::iterator it = pDevice->channel_list_.begin();
            for( ; it!=pDevice->channel_list_.end(); ++it )
            {
                if( it->channel_status == protocol::CHANNEL_STS_ONLINE 
                    && it->channel_id <= pDevice->channel_num_ )
                {
                    int pos = it->channel_id - 1;
                    channel_status.replace( pos, 1, 1, '1');
                }
            }
            reply["channel_status"] = channel_status;
        }
        else
        {
            reply["status"] = "offline";
            reply["timestamp"] = TimestampStr(pDevice->timestamp_);
        }

    }while(false);

    return true;
}

void HttpRequestHandler::MakeHttpResponse(const std::string& content, SHttpResponsePara_ptr& http_resp)
{
    http_resp->pContent.reset(new uint8[content.length() + 1]);
    memcpy(http_resp->pContent.get(),content.c_str(),content.length() + 1);    
    http_resp->content_len = content.length();    
    http_resp->content_type = "application/json"; // text/html
    http_resp->ret_code = "200 OK";  
}

int HttpRequestHandler::NotHandleOk(SHttpResponsePara_ptr& http_resp)
{
    Variant reply;
    reply["code"] = -1;
    reply["message"] = "Invalid Parameter";
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content, http_resp);
    return -1;
}

int HttpRequestHandler::HttpNotFound(SHttpResponsePara_ptr& http_resp)
{
    http_resp->ret_code = "404 Not Found";
    http_resp->keep_alive = false;
    return -1;
}