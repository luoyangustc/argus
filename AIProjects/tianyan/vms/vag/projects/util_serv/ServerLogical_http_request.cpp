#include "ServerLogical.h"

#include <sstream>
#include "http_header_util.h"
#include "util_module.h"
#include "AYServerApi.h"
#include "variant.h"

int CServerLogical::DefaultHandle(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    const std::string content = 
        "<html><title>result</title><body>Welcome to Ulucu util Server world !</body></html>";

    MakeHttpResponse(content, pRes);
    return 0;  // ok  
}

int CServerLogical::GenerateDeviceId(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{  
    HttpHeaderIterator iter = params.find("oem_id");
    if (iter == params.end())
    {
        Warn("oem_id missing\n");
        return -1;
    }
    // step1 : copy oem_id
    char original_did[24] = "";  // original_did's length is 18 bytes  
    const int oem_id_length = 2;  // oem_id's length is 2 bytes  
    if (oem_id_length == iter->second.length())  
    {
        memcpy(original_did,iter->second.c_str(),oem_id_length);
    }
    else
    {
        Warn("oem_id length != 2 bytes\n");
        return -1;
    }

    // step2 : copy sn  
    iter = params.find("sn");
    if (iter == params.end())
    {
        Warn("sn missing\n");
        return -1;    
    }
    const int sn_max_length = 16;  // sn's max length is 16bytes
    if (iter->second.length() <= sn_max_length)
    {
        if (iter->second.length() == sn_max_length)
        {
            memcpy(original_did + oem_id_length, iter->second.c_str(), sn_max_length);        
        }
        else
        {
            const int zero_num = sn_max_length - iter->second.length();
            for (int index = 0; index < zero_num; ++index)
            {
                // If the number is not enough, then fill character 'F'
                original_did[oem_id_length + index] = '0';  // FIXME : F
            }
            memcpy(original_did + oem_id_length + zero_num, iter->second.c_str(), iter->second.length());
        }  
    }
    else
    {
        Warn("sn length > 16 bytes\n");
        return -1;
    }

    Debug("original_did is %s \n",original_did);  

    char generate_did[32] = "";
    unsigned int length = sizeof generate_did;
    Variant reply;
    int ok = -1; 
    if (um_generate_device_id(original_did, generate_did, &length) < 0)
    {
        reply["code"] = kServerInternalError;
        reply["message"] = "Server Internal Error";
    }
    else
    {
        Variant data;
        data["device_id"] = generate_did;
        
        reply["code"]= kErrorNone;
        reply["message"] = "Success";
        reply["data"] = data;

        ok = 0;
    }
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);
    return ok; 
}

int CServerLogical::CheckDeviceId(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    HttpHeaderIterator iter = params.find("device_id");
    if (iter == params.end())
    {
        Warn("device_id missing\n");   
        return -1;
    }
    std::string device_id = iter->second;

    int max_group = -1;  
    iter = params.find("max_group");
    if (iter == params.end())
    {
        Warn("max_group missing\n");  //optionnal
    }
    else
    {
        max_group = atoi(iter->second.c_str());
    }

    bool is_valid = true;
    if (um_is_valid_device_id(device_id.c_str()) < 0)
    {
        Warn("%s is invalid \n",device_id.c_str());  
        is_valid = false;
    } 

    Variant reply;
    if (max_group > 0)
    {
        int group_id = 1;  // FIXME
        if ((group_id = um_get_group(device_id.c_str(), max_group)) < 0)
        {
            Warn("um_get_group failed \n");
            reply["code"] = kServerInternalError;
            reply["message"] = "Server Internal Error";
        }
        else
        {
            reply["code"] = kErrorNone;
            reply["is_valid"] = is_valid ? 1 : 0;      
            reply["group_id"] = group_id;     
        } 
    }
    else
    {
        reply["code"] = kErrorNone;
        reply["is_valid"] = is_valid ? 1 : 0;    
    }
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);
    return 0;
}

int CServerLogical::GenerateDeviceToken(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    token_source_t token_source;
    memset(&token_source, 0x0, sizeof(token_source));

    HttpHeaderIterator iter = params.find("device_id");
    if (iter == params.end())
    {
        Warn("device_id missing\n");   
        return -1;
    }
    token_source.mask = 0x08;
    memcpy(token_source.device_id, iter->second.c_str(), iter->second.length() + 1);

    iter = params.find("duration");
    if (iter == params.end())
    {
        Warn("duration missing\n");  //optionnal
        return -1;
    }
    token_source.mask |= 0x01;
    token_source.duration_time = atoi(iter->second.c_str());

    iter = params.find("channel_idx");
    if (iter == params.end())
    {
        Warn("channel_idx missing\n");  //optionnal
    }
    else
    {
        token_source.channel_idx = atoi(iter->second.c_str());
    }

    iter = params.find("public_ip");
    if (iter == params.end())
    {
        Warn("public_ip missing\n");  //optionnal
    }
    else
    {
        token_source.mask |= 0x02;
        memcpy(token_source.client_ip, iter->second.c_str(), iter->second.length() + 1);
    }  
    iter = params.find("public_port");
    if (iter == params.end())
    {
        Warn("public_port missing\n");  //optionnal
    }
    else
    {
        token_source.client_port = atoi(iter->second.c_str());
    }

    iter = params.find("private_ip");
    if (iter == params.end())
    {
        Warn("private_ip missing\n");  //optionnal
    }
    else
    {
        token_source.mask |= 0x04;
        memcpy(token_source.client_private_ip, iter->second.c_str(), iter->second.length() + 1);
    }

    iter = params.find("private_port");
    if (iter == params.end())
    {
        Warn("private_port missing\n");  //optionnal
    }
    else
    {
        token_source.client_private_port = atoi(iter->second.c_str());
    }

    iter = params.find("version");
    if (iter == params.end())
    {
        Warn("version missing\n");  //optionnal    
    }
    else
    {
        token_source.mask |= 0x20;
        memcpy(token_source.version, iter->second.c_str(), iter->second.length() + 1);
    }

    iter = params.find("factory");
    if (iter == params.end())
    {
        Warn("factory missing\n");  //optionnal
    }
    else
    {
        token_source.mask |= 0x80;
        memcpy(token_source.factory, iter->second.c_str(), iter->second.length() + 1);
    }

    iter = params.find("customer");
    if (iter == params.end())
    {
        Warn("customer missing\n");  //optionnal;
    }
    else
    {
        token_source.mask |= 0x100;
        memcpy(token_source.customer, iter->second.c_str(), iter->second.length() + 1);
    }

    iter = params.find("server_ip");
    if (iter == params.end())
    {
        Warn("server_ip missing\n");  //optionnal
    }
    else
    {
        token_source.mask |= 0x200;
        memcpy(token_source.serv_ip, iter->second.c_str(), iter->second.length() + 1);  
    }

    iter = params.find("server_port");
    if (iter == params.end())
    {
        Warn("server_port missing\n");  //optionnal
    }
    else
    {
        token_source.serv_port = atoi(iter->second.c_str());
    }

    const int buffer_size = 1024;  // FIXME : may be enough
    char token[buffer_size] = "";
    unsigned int token_length = sizeof token;
    bool ok = true;
    if (um_generate_token(&token_source, token, &token_length) < 0)
    {
        Debug("not generateing token \n");
        ok = false;
    }

    // make json content
    Variant reply;
    if (ok)
    {
        reply["code"] = kErrorNone;
        reply["token"] = token;
    }
    else
    {
        reply["code"] = kServerInternalError;
        reply["message"] = "Server Internal Error"; 
    }

    std::string content;
    reply.SerializeToJSON(content); 
    MakeHttpResponse(content,pRes);

    Debug("response content:\n%s \n",content.c_str());

    return 0;
}

int CServerLogical::DecryptDeviceToken(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    HttpHeaderIterator iter = params.find("token");
    if (iter == params.end())
    {
        Warn("token missing\n");   
        return -1;
    }

    token_source_t token_source;
    memset(&token_source, 0x0, sizeof(token_source));

    bool ok = true;
    if (um_decrypt_token(iter->second.c_str(), &token_source) < 0)
    {
        Warn("decrypt token error \n");
        ok = false;
    }

    Variant reply;
    if (ok)
    {
        reply["code"] = kErrorNone;
        if (token_source.mask & 0x01)
        {
            reply["duration"] = token_source.duration_time;
        }
        if (token_source.mask & 0x02)
        {
            reply["public_ip"] = token_source.client_ip;
            reply["public_port"] = token_source.client_port;
        }
        if (token_source.mask & 0x04)
        {
            reply["private_ip"] = token_source.client_private_ip;
            reply["private_port"] = token_source.client_private_port;
        }
        if (token_source.mask & 0x08)
        {
            reply["device_id"] = token_source.device_id;
        }
        if (token_source.mask & 0x10)
        {
            reply["user_name"] = token_source.user_name;
        }
        if (token_source.mask & 0x20)
        {
            reply["version"] = token_source.version;
        }
        if (token_source.mask & 0x40)
        {
            reply["cookie"] = token_source.cookie;
        }
        if (token_source.mask & 0x80)
        {
            reply["factory"] = token_source.factory;
        }
        if (token_source.mask & 0x100)
        {
            reply["customer"] = token_source.customer;
        }
        if (token_source.mask & 0x200)
        {
            reply["server_ip"] = token_source.serv_ip;
            reply["server_port"] = token_source.serv_port;
        }     
    }
    else
    {
        reply["code"] = kServerInternalError;
        reply["message"] = "Server Internal Error";
    } 
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);
    return 0;
}

int CServerLogical::GenerateUserToken(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    token_source_t token_source;
    memset(&token_source, 0x0, sizeof(token_source));

    HttpHeaderIterator iter = params.find("flag");
    if (iter == params.end())
    {
        Warn("flag missing\n");   
        return -1;
    }  
    token_source.flag = atoi(iter->second.c_str());  // flag

    iter = params.find("user_name");
    if (iter == params.end())
    {
        Warn("user_name missing\n");   
        return -1;
    }
    token_source.mask = 0x10;
    memcpy(token_source.user_name, iter->second.c_str(), iter->second.length() + 1);

    iter = params.find("device_id");
    if (iter == params.end())
    {
        Warn("device_id missing\n");   
        return -1;
    }
    token_source.mask |= 0x08;
    memcpy(token_source.device_id, iter->second.c_str(), iter->second.length() + 1);

    iter = params.find("duration");
    if (iter == params.end())
    {
        Warn("duration missing\n");  //optionnal
        return -1;
    }
    token_source.mask |= 0x01;
    token_source.duration_time = atoi(iter->second.c_str());

    iter = params.find("channel_idx");
    if (iter == params.end())
    {
        Warn("channel_idx missing\n");  //optionnal
    }
    else
    {
        token_source.channel_idx = atoi(iter->second.c_str());
    }

    iter = params.find("public_ip");
    if (iter == params.end())
    {
        Warn("public_ip missing\n");  //optionnal
    }
    else
    {
        token_source.mask |= 0x02;
        memcpy(token_source.client_ip, iter->second.c_str(), iter->second.length() + 1);
    }  

    iter = params.find("public_port");
    if (iter == params.end())
    {
        Warn("public_port missing\n");  //optionnal
    }
    else
    {  
        token_source.client_port = atoi(iter->second.c_str());
    }

    iter = params.find("private_ip");
    if (iter == params.end())
    {
        Warn("private_ip missing\n");  //optionnal
    }
    else
    {  
        token_source.mask |= 0x04;
        memcpy(token_source.client_private_ip, iter->second.c_str(), iter->second.length() + 1);  
    }

    iter = params.find("private_port");
    if (iter == params.end())
    {
        Warn("private_port missing\n");  //optionnal
    }
    else
    {   
        token_source.client_private_port = atoi(iter->second.c_str());
    }

    iter = params.find("version");
    if (iter == params.end())
    {
        Warn("version missing\n");  //optionnal    
    }
    else
    {  
        token_source.mask |= 0x20;
        memcpy(token_source.version, iter->second.c_str(), iter->second.length() + 1);
    }

    iter = params.find("factory");
    if (iter == params.end())
    {
        Warn("factory missing\n");  //optionnal
    }
    else
    {  
        token_source.mask |= 0x80;
        memcpy(token_source.factory, iter->second.c_str(), iter->second.length() + 1);  
    }

    iter = params.find("customer");
    if (iter == params.end())
    {
        Warn("customer missing\n");  //optionnal;
    }
    else
    {  
        token_source.mask |= 0x100;
        memcpy(token_source.customer, iter->second.c_str(), iter->second.length() + 1);  
    }

    iter = params.find("server_ip");
    if (iter == params.end())
    {
        Warn("server_ip missing\n");  //optionnal
    }
    else
    {  
        token_source.mask |= 0x200;
        memcpy(token_source.serv_ip, iter->second.c_str(), iter->second.length() + 1);  
    }

    iter = params.find("server_port");
    if (iter == params.end())
    {
        Warn("server_port missing\n");  //optionnal
    }
    else
    {
        token_source.serv_port = atoi(iter->second.c_str());
    }

    const int buffer_size = 1024;  // FIXME : may be enough
    char token[buffer_size] = "";
    unsigned int token_length = sizeof token;
    bool ok = true;
    if (um_generate_token(&token_source, token, &token_length) < 0)
    {
        Error("not generate token \n");
        ok = false;
    }

    // json data
    Variant reply;
    if (ok)
    {
        reply["code"] = kErrorNone;
        reply["token"] = token;
    }
    else
    {
        reply["code"] = kServerInternalError;
        reply["message"] = "Server Internal Error";
    } 
    std::string content;
    reply.SerializeToJSON(content); 
    MakeHttpResponse(content,pRes);

    Debug("response content:\n%s \n", content.c_str());

    return 0;
}

int CServerLogical::DecryptUserToken(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    HttpHeaderIterator iter = params.find("token");
    if (iter == params.end())
    {
        Warn("token missing\n");   
        return -1;
    }

    token_source_t token_source;
    memset(&token_source, 0x0, sizeof(token_source));

    bool ok = true;
    if (um_decrypt_token(iter->second.c_str(), &token_source) < 0)
    {
        Error("decrypt token error \n");
        ok = false;
    }
    Debug("decrypt user token ok \n");

    Variant reply;
    if (ok)
    {
        reply["code"] = kErrorNone;
        reply["flag"] = token_source.flag;
        if (token_source.mask & 0x01)
        {
            reply["duration"] = token_source.duration_time;
        }
        if (token_source.mask & 0x02)
        {
            reply["public_ip"] = token_source.client_ip;
            reply["public_port"] = token_source.client_port;
        }
        if (token_source.mask & 0x04)
        {
            reply["private_ip"] = token_source.client_private_ip;
            reply["private_port"] = token_source.client_private_port;
        }
        if (token_source.mask & 0x08)
        {
            reply["device_id"] = token_source.device_id;  // FIXME : channel_idx
        }
        if (token_source.mask & 0x10)
        {
            reply["user_name"] = token_source.user_name;
        }
        if (token_source.mask & 0x20)
        {
            reply["version"] = token_source.version;
        }
        if (token_source.mask & 0x40)
        {
            reply["cookie"] = token_source.cookie;
        }
        if (token_source.mask & 0x80)
        {
            reply["factory"] = token_source.factory;
        }
        if (token_source.mask & 0x100)
        {
            reply["customer"] = token_source.customer;
        }
        if (token_source.mask & 0x200)
        {
            reply["server_ip"] = token_source.serv_ip;
            reply["server_port"] = token_source.serv_port;
        }     
    }
    else
    {
        reply["code"] = kServerInternalError;
        reply["message"] = "Server Internal Error";
    } 
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);
    return 0;
}

int CServerLogical::EncryptString(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    HttpHeaderIterator iter = params.find("algo");
    if (iter == params.end())
    {
        Warn("token missing\n");   
        return -1;
    }
    std::string algo = iter->second;

    iter = params.find("plain_str");
    if (iter == params.end())
    {
        Warn("token missing\n");   
        return -1;
    }
    std::string plain_str = iter->second;

    std::string key = "Ulucu888";  // default key
    iter = params.find("key");
    if (iter == params.end())
    {
        Warn("token missing\n");  // optional
    }
    else
    {
        key = iter->second;
    }

    char encrypt_str[1024] = "";
    unsigned int length = sizeof encrypt_str;
    bool ok = true;
    const std::string kDesSafeBase64 = "1";
    const std::string kDeshex = "2";
    const std::string kSafeBase64 = "3";
    if (algo == kDesSafeBase64)  // 1 : des_safebase64
    {
        // FIXME : not using key !
        if (um_encrypt_string(plain_str.c_str(), enm_algo_des_safe_base64, key.c_str(), encrypt_str, &length) < 0)
        {
            ok = false;
        }
    }
    else if (algo == kDeshex)  // 2 : des_hex
    {    
        if (um_encrypt_string(plain_str.c_str(), enm_algo_des_hex, key.c_str(), encrypt_str, &length) < 0)
        {
            ok = false;
        }
    }
    else if (algo == kSafeBase64)
    {
        if (um_encrypt_string(plain_str.c_str(), enm_algo_safe_base64, key.c_str(), encrypt_str, &length) < 0)
        {
            ok = false;
        }  
    }

    Variant reply;
    if (ok)
    {
        reply["code"] = kErrorNone;
        reply["encrypted_str"] = encrypt_str;
    }
    else
    {
        reply["code"] = kServerInternalError;
        reply["message"] = "Server Internal Error";
    } 
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);
    return 0;
}

int CServerLogical::DecryptString(HttpHeaderMap& params,SHttpResponsePara_ptr& pRes)
{
    HttpHeaderIterator iter = params.find("algo");
    if (iter == params.end())
    {
        Warn("token missing\n");   
        return -1;
    }
    std::string algo = iter->second;

    iter = params.find("encrypted_str");
    if (iter == params.end())
    {
        Warn("token missing\n");   
        return -1;
    }
    std::string encrypted_str = iter->second;

    std::string key = "Ulucu888";  // default key
    iter = params.find("key");
    if (iter == params.end())
    {
        Warn("token missing\n");  // optional
    }
    else
    {
        key = iter->second;
    }

    char plain_str[1024] = "";
    unsigned int length = sizeof plain_str;
    bool ok = true;
    const std::string kDesSafeBase64 = "1";
    const std::string kDeshex = "2";
    const std::string kSafeBase64 = "3";
    if (algo == kDesSafeBase64)  // 1 : des_zbase
    {
        // FIXME : not using key
        if (um_decrypt_string(encrypted_str.c_str(), enm_algo_des_safe_base64, key.c_str(), plain_str, &length) < 0)
        {
            ok = false;
        }
    }
    else if (algo == kDeshex)  // 2 : des_hex
    {    
        if (um_decrypt_string(encrypted_str.c_str(), enm_algo_des_hex, key.c_str(), plain_str, &length) < 0)
        {
            ok = false;
        }
    }
    else if (algo == kSafeBase64)
    {
        if (um_decrypt_string(encrypted_str.c_str(),enm_algo_safe_base64,key.c_str(),plain_str,&length) < 0)
        {
            ok = false;
        }
    }

    Variant reply;
    if (ok)
    {
        reply["code"] = kErrorNone;
        reply["plain_str"] = plain_str;
    }
    else
    {
        reply["code"] = kServerInternalError;
        reply["message"] = "Server Internal Error";
    } 
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);
    return 0;
}

int CServerLogical::GetAccessToken(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes)
{
    Variant reply;
    {
        Variant data;
        data["access_token"] = "1fb72cd23620be50f8460fd4f15664da9dee63c2";

        reply["code"]= kErrorNone;
        reply["message"] = "Success";
        reply["data"] = data;
    }

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);

    return 0; 
}

int CServerLogical::GetAccessServer(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes)
{
    Variant reply;
    {
        Variant data;
        data["access_server_ip"] = "100.100.60.192";
        data["access_server_port"] = "9100";

        reply["code"]= kErrorNone;
        reply["message"] = "Success";
        reply["data"] = data;
    }

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);

    return 0; 
}

int CServerLogical::StreamRRS(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes)
{
    Variant reply;
    {
        Variant server;
        {
            Variant stream = "100.100.60.192:9200";
            server.PushToArray(stream);
        }
        

        Variant data;
        data["server"] = server;

        reply["code"]= kErrorNone;
        reply["message"] = "Success";
        reply["data"] = data;
    }

    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content,pRes);

    return 0; 
}

int CServerLogical::HttpNotFound(SHttpResponsePara_ptr& httpResp)
{
    httpResp->ret_code = "404 Not Found";
    httpResp->keep_alive = false;
    return -1;
}

void CServerLogical::MakeHttpResponse(const std::string& content, SHttpResponsePara_ptr& pRes)
{
    pRes->pContent.reset(new uint8[content.length() + 1]);
    memcpy(pRes->pContent.get(),content.c_str(),content.length() + 1);    
    pRes->content_len = content.length();    
    pRes->content_type = "text/html";
    pRes->ret_code = "200 OK";  
}

void CServerLogical::NotHandleOk(SHttpResponsePara_ptr& pRes)
{
    Variant reply;
    reply["code"] = kInvalidParameter;
    reply["message"] = "Invalid Parameter";
    std::string content;
    reply.SerializeToJSON(content);

    MakeHttpResponse(content, pRes);
}

int CServerLogical::HandleHttpRequest(const std::string& page,
    HttpHeaderMap& params,
    SHttpResponsePara_ptr& pRes)
{
    Debug("page is %s \n",page.c_str());      

    const std::string root_directory = "/";
    const std::string gen_device_id = "/gen_device_id";
    const std::string check_device_id = "/check_device_id";
    const std::string gen_device_token = "/gen_device_token";
    const std::string decrypt_device_token = "/decrypt_device_token";
    const std::string gen_user_token = "/gen_user_token";
    const std::string decrypt_user_token = "/decrypt_user_token";
    const std::string encrypt_string = "/encrypt_string";
    const std::string decrypt_string = "/decrypt_string";
    const std::string get_access_serv = "/get_access_server";
    const std::string get_access_token = "/get_access_token";
    const std::string streamrrs = "/streamrrs";
    if (page == root_directory)
    {
        return DefaultHandle(params,pRes);
    }
    else if (page == gen_device_id)
    {
        return GenerateDeviceId(params,pRes);
    }
    else if (page == check_device_id)
    {
        return CheckDeviceId(params,pRes);
    }
    else if (page == gen_device_token)
    {
        return GenerateDeviceToken(params,pRes);
    }
    else if (page == decrypt_device_token)
    {
        return DecryptDeviceToken(params,pRes);
    }
    else if (page == gen_user_token)
    {
        return GenerateUserToken(params,pRes);
    }
    else if (page == decrypt_user_token)
    {
        return DecryptUserToken(params,pRes);    
    }
    else if (page == encrypt_string)
    {
        return EncryptString(params,pRes);
    }
    else if (page == decrypt_string)
    {
        return DecryptString(params,pRes);
    }
    else if (page == get_access_serv)
    {
        return GetAccessServer(params,pRes);
    }
    else if (page == get_access_token)
    {
        return GetAccessToken(params,pRes);
    }
    else if (page == streamrrs)
    {
        return StreamRRS(params,pRes);
    }
    else
    {
        return HttpNotFound(pRes);
    }
}

int32 CServerLogical::OnHttpClientRequest(ITCPSessionSendSink*sink,
    CHostInfo& hiRemote,
    SHttpRequestPara_ptr pReq,
    SHttpResponsePara_ptr pRes)
{
    BOOST_ASSERT(pReq);
    BOOST_ASSERT(pRes);

    (void)sink;

    do 
    {
        if (! pReq->header_detail->is_request) break;    

        Debug("from(%s), http request url(%s)",hiRemote.GetNodeString().c_str(),
            pReq->header_detail->url_.c_str());

        CHttpHeader_ptr httpHeader = pReq->header_detail;
        if (HandleHttpRequest(httpHeader->url_detail_.page_, httpHeader->url_detail_.params_, pRes) < 0)
        {
            NotHandleOk(pRes);
        }

        return pReq->header_data_len;  // return http request length
    } while(false);

    return -1;
}
