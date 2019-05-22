
#ifndef _WINDOWS
#include <arpa/inet.h>
#else
#include <WinSock2.h>
#endif
#include <time.h>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp> 
#include <openssl/hmac.h>
#include <openssl/engine.h>
#include <openssl/evp.h>
#include "http_header_util.h"
#include "DeviceID.h"
#include "ZBase64.h"
#include "logging_posix.h"
#include "TokenMgr.h"


CTokenMgr::CTokenMgr(void)
{
}

CTokenMgr::~CTokenMgr(void)
{
}

void CTokenMgr::Update()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    uint32 current_time = time(NULL);
    TokenMap::right_iterator it = tokens_.right.begin();

    for (; it != tokens_.right.end(); )
    {
        if (it->first <= current_time)
        {
            if( get_current_tick() - it->second.create_tick > 12*3600*1000 )
            {
                tokens_.right.erase(it++);
            }
            else
            {
                break;
            }			
        }
        else
        {
            break;
        }
    }
}

bool CTokenMgr::SetKey(const string& access_key, const string& secret_key)
{
    if(access_key.empty() || secret_key.empty())
    {
        Error("erro, access_key(%s), secret_key(%s)!",access_key.c_str(),secret_key.c_str());
        return false;
    }

    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        key_tables_[access_key] = secret_key;
    }
    Debug("set key success, access_key(%s), secret_key(%s)!",access_key.c_str(),secret_key.c_str());
    return true;
}

bool CTokenMgr::TokenGen(IN const string& access_key, IN const string& plain_text, OUT string& token)
{
    do 
    {
        //通过公钥找私钥
        string secret_key = "";
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<string,string>::iterator it = key_tables_.find(access_key);
            if ( it == key_tables_.end() )
            {
                Error("find secret key failed by access key(%s)!",access_key.c_str());
                break;
            }
            secret_key = it->second;
        }

        //生成密文
        string cipher_text="";
        {
            HMAC_CTX hmac_ctx;
            unsigned char digest[EVP_MAX_MD_SIZE + 1];
            unsigned int digest_len = sizeof(digest);
            HMAC_CTX_init(&hmac_ctx);
            HMAC_Init_ex(&hmac_ctx, secret_key.c_str(), secret_key.length(), EVP_sha1(), NULL);
            HMAC_Update(&hmac_ctx, (unsigned char*)plain_text.c_str(), plain_text.length());
            HMAC_Final(&hmac_ctx, digest, &digest_len);
            HMAC_CTX_cleanup(&hmac_ctx);
            cipher_text = ZBase64::Safe_Encode(digest, digest_len);
        }

        //明文safe_base64加密
        string plain_text_b64="";
        {
            plain_text_b64 = ZBase64::Safe_Encode((const unsigned char*)plain_text.c_str(), plain_text.length());
        }

        //合成token
        {
            char szToke[256+1]={0};
            int len = sprintf(szToke, "%s:%s:%s", access_key.c_str(), cipher_text.c_str(), plain_text_b64.c_str());
            if(len <0)
            {
                Error("bulid token failed, (%s:%s:%s)!",access_key.c_str(),cipher_text.c_str(), plain_text_b64.c_str());
                break;
            }
            szToke[len] = '\0';

            token = szToke;
        }
        return true;
    } while (0);
    return false;
}

bool CTokenMgr::TokenVerify(IN const string& token, IN const string& token_type, OUT string& meta_data)
{
    do
    {
        //获取公钥、密文、明文
        string access_key="";    //公钥
        string cipher_text="";   //密文
        string plain_text="";    //明文
        string plain_text_b64="";
        {
            boost::cmatch mat;
            boost::regex reg( "(.+):(.+):(.+)" );
            bool r = boost::regex_match( token.c_str(), mat, reg);
            if(!r)
            {
                Error("token incorret, token(%s)!",token.c_str());
                break;
            }

            access_key= mat[1].str();
            cipher_text= mat[2].str();
            plain_text_b64 = mat[3].str();

            char tmp_buf[128+1] = {0};
            int len = sizeof(tmp_buf);
            if(ZBase64::Safe_Decode(plain_text_b64.c_str(), plain_text_b64.length(), (unsigned char*)tmp_buf, len) <= 0)
            {
                Error("plain text safe base64 decode failed, token(%s)!",token.c_str());
                break;
            }
            plain_text.assign(tmp_buf, len);
        }

        //明文解析,并检查是否过期
        {
            //get expire time
            std::size_t pos1 = plain_text.find_first_of(':');
            if(pos1==std::string::npos)
            {
                Error("plain text is incorrect, (%s)!",plain_text.c_str());
                break;
            }
            string s_expire_time(plain_text, 0, pos1-0);
            uint32 expire_time = atoi(s_expire_time.c_str());
            uint32 current_time = time(NULL);
            if( current_time > expire_time)
            {
                Error("token is expire, (%u>%u)!",current_time, expire_time);
                break;
            }
            
            //get token type
            std::size_t pos2 = plain_text.find_first_of(':', pos1+1);
            if(pos2==std::string::npos)
            {
                Error("plain text is incorrect, (%s)!",plain_text.c_str());
                break;
            }
            string s_token_type(plain_text, pos1+1, pos2-pos1-1);
            if( token_type != s_token_type )
            {
                Error("token type is incorrect, (%s!=%s)!",token_type.c_str(), s_token_type.c_str());
                break;
            }
            
            //get meta data
            meta_data.assign(plain_text, pos2+1, string::npos);
        }

        //通过公钥找私钥
        string secret_key = "";
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<string,string>::iterator it = key_tables_.find(access_key);
            if ( it == key_tables_.end() )
            {
                Error("find secret key failed by access key(%s)!",access_key.c_str());
                break;
            }
            secret_key = it->second;
        }
        
        HMAC_CTX hmac_ctx;
        unsigned char digest[EVP_MAX_MD_SIZE + 1];
        unsigned int digest_len = sizeof(digest);
        HMAC_CTX_init(&hmac_ctx);
        HMAC_Init_ex(&hmac_ctx, secret_key.c_str(), secret_key.length(), EVP_sha1(), NULL);
        HMAC_Update(&hmac_ctx, (unsigned char*)plain_text.c_str(), plain_text.length());
        HMAC_Final(&hmac_ctx, digest, &digest_len);
        HMAC_CTX_cleanup(&hmac_ctx);
        string sign_text = ZBase64::Safe_Encode(digest, digest_len);
        if(sign_text!=cipher_text)
        {
            Error("cipher text is auth failed, (%s != %s)!",sign_text.c_str(), cipher_text.c_str());
            break;
        }
        return true;
    } while (0);    
    return false;
}

//会话服务token
bool CTokenMgr::SessionDeviceToken_Auth( IN const string& token, IN const string& device_id )
{
    do 
    {
        string meta_data;
        if(!TokenVerify(token, "ulu_device_session", meta_data))
        {
            break;
        }
        if(meta_data != device_id)
        {
            Error("device id check failed, (%s!=%s)!",meta_data.c_str(), device_id.c_str());
            break;
        }
        return true;
    } while (0);
    return false;
}

bool CTokenMgr::SessionUserToken_Auth( IN const string& token, IN const string& user_id )
{
    do 
    {
        string meta_data;
        if(!TokenVerify(token, "ulu_user_session", meta_data))
        {
            break;
        }
        if(meta_data != user_id)
        {
            Error("user id check failed, (%s!=%s)!",meta_data.c_str(), user_id.c_str());
            break;
        }
        return true;
    } while (0);
    return false;
}

bool CTokenMgr::SessionDeviceToken_Gen( IN const string& device_id, IN uint32 duration, IN const string& access_key, OUT string& token )
{
    do 
    {
        string toke_type("ulu_device_session");
        uint32 expire = time(NULL)+duration;
        char szplain_text[128+1]={0};
        int len = snprintf( szplain_text, sizeof(szplain_text)-1, "%u:%s:%s", expire, toke_type.c_str(), device_id.c_str() );
        if(len <0)
        {
            Error("bulid plain text failed, (%u:%s:%s)!", expire, toke_type.c_str(), device_id.c_str());
            break;
        }
        return TokenGen(access_key, szplain_text, token);
    } while (0);
    return false;
}

bool CTokenMgr::SessionUserToken_Gen( IN const string& user_id, IN uint32 duration, IN const string& access_key, OUT string& token )
{
    do 
    {
        string toke_type("ulu_user_session");
        uint32 expire = time(NULL)+duration;
        char szplain_text[128+1]={0};
        int len = snprintf( szplain_text, sizeof(szplain_text)-1, "%u:%s:%s", expire, toke_type.c_str(), user_id.c_str() );
        if(len <0)
        {
            Error("bulid plain text failed, (%u:%s:%s)!", expire, toke_type.c_str(), user_id.c_str());
            break;
        }
        return TokenGen(access_key, szplain_text, token);
    } while (0);
    return false;
}

//流服务token
bool CTokenMgr::StreamDeviceToken_Auth( IN const string& token, IN const string& device_id, IN int channel_id )
{
    do 
    {
        string meta_data;
        if(!TokenVerify(token, "ulu_device_stream", meta_data))
        {
            break;
        }

        {
            boost::cmatch mat;
            boost::regex reg( "(.+):([0-9]+)" );
            bool r = boost::regex_match( meta_data.c_str(), mat, reg);
            if(!r)
            {
                Error("meta data is incorret, (%s)!",meta_data.c_str());
                break;
            }

            if( device_id != mat[1].str() )
            {
                Error("device id check failed, (%s!=%s)!",device_id.c_str(), mat[1].str().c_str());
                break;
            }

            int t_channel_id = boost::lexical_cast<int>(mat[2].str().c_str());
            if( channel_id > t_channel_id)
            {
                Error("channel id check failed, (%u!=%u)!",channel_id, t_channel_id);
                break;
            }
        }
        return true;
    } while (0);
    return false;
}

bool CTokenMgr::StreamUserToken_Auth( IN const string& token, IN const string& user_id, IN const string& device_id, IN int channel_id )
{
    do
    {
        string meta_data;
        if(!TokenVerify(token, "ulu_user_stream", meta_data))
        {
            break;
        }

        {
            boost::cmatch mat;
            boost::regex reg( "(.+):([0-9]+):(.+)" );
            bool r = boost::regex_match( meta_data.c_str(), mat, reg);
            if(!r)
            {
                Error("meta data is incorret, (%s)!",meta_data.c_str());
                break;
            }

            if( device_id != mat[1].str() )
            {
                Error("device id check failed, (%s!=%s)!",device_id.c_str(), mat[1].str().c_str());
                break;
            }

            int t_channel_id = boost::lexical_cast<int>(mat[2].str().c_str());
            if( channel_id > t_channel_id)
            {
                Error("channel id check failed, (%u!=%u)!",channel_id, t_channel_id);
                break;
            }

            if( user_id != mat[3].str() )
            {
                Error("user id check failed, (%s!=%s)!",user_id.c_str(), mat[3].str().c_str());
                break;
            }
        }
        return true;
    } while (0);
    return false;
}

bool CTokenMgr::StreamDeviceToken_Gen( IN const string& device_id, IN int channel_id, IN uint32 duration, IN const string& access_key, OUT string& token )
{
    do 
    {
        string toke_type("ulu_device_stream");
        uint32 expire = time(NULL)+duration;
        char szplain_text[128+1]={0};
        int len = snprintf( szplain_text, sizeof(szplain_text)-1, "%u:%s:%s:%d", expire, toke_type.c_str(), device_id.c_str(), channel_id );
        if(len <0)
        {
            Error("bulid plain text failed, (%u:%s:%s:%d)!", expire, toke_type.c_str(), device_id.c_str(), channel_id);
            break;
        }
        return TokenGen(access_key, szplain_text, token);
    } while (0);
    return false;
}

bool CTokenMgr::StreamUserToken_Gen( IN const string& user_id, IN const string& device_id, IN int channel_id, IN uint32 duration, IN const string& access_key, OUT string& token )
{
    do 
    {
        string toke_type("ulu_user_stream");
        uint32 expire = time(NULL)+duration;
        char szplain_text[128+1]={0};
        int len = snprintf( szplain_text, sizeof(szplain_text)-1, "%u:%s:%s:%d:%s", expire, toke_type.c_str(), device_id.c_str(), channel_id, user_id.c_str() );
        if(len <0)
        {
            Error("bulid plain text failed, (%u:%s:%s:%d:%s)!", expire, toke_type.c_str(), device_id.c_str(), channel_id, user_id.c_str());
            break;
        }
        return TokenGen(access_key, szplain_text, token);
    } while (0);
    return false;
}


bool CTokenMgr::Test()
{
    string access_key = "4_odedBxmrAHiu4Y0Qp0HPG0NANCf6VAsAjWL_k9";
    string secret_key = "SrRuUVfDX6drVRvpyN8mv8Vcm9XnMZzlbDfvVfMe";
    SetKey(access_key, secret_key);

    string user_id ="zhangysh";
    string device_id = "Ub0000000684216279FY";
    int channel_id = 1;

    do 
    {
        {
            string out_token;
            if ( !SessionDeviceToken_Gen(device_id, 3600, access_key, out_token) )
            {
                printf("gen device session token failed!");
                break;
            }
            if(!SessionDeviceToken_Auth(out_token, device_id))
            {
                printf("check device session token failed, token(%s)!", out_token.c_str());
                break;
            }
        }
        
        {
            string out_token;
            if ( !SessionUserToken_Gen(user_id, 3600, access_key, out_token) )
            {
                printf("gen user session token failed!");
                break;
            }
            if(!SessionUserToken_Auth(out_token, user_id))
            {
                printf("check user session token failed, token(%s)!", out_token.c_str());
                break;
            }
        }

        {
            string out_token;
            if ( !StreamDeviceToken_Gen(device_id, channel_id, 3600, access_key, out_token) )
            {
                printf("gen device stream token failed!");
                break;
            }
            if(!StreamDeviceToken_Auth(out_token, device_id, channel_id))
            {
                printf("check device stream token failed, token(%s)!", out_token.c_str());
                break;
            }
        }

        {
            string out_token;
            if ( !StreamUserToken_Gen(user_id, device_id, channel_id, 3600, access_key, out_token) )
            {
                printf("gen user stream token failed!");
                break;
            }
            if(!StreamUserToken_Auth(out_token, user_id, device_id, channel_id))
            {
                printf("check user stream token failed, token(%s)!", out_token.c_str());
                break;
            }
        }
        
        printf("test success!");
        return true;
    } while (0);
    printf("test failed!");
    return false;
}
