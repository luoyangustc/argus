#include "logging_posix.h"
#include "Config.h"

CStreamCfg::CStreamCfg()
    : enable_token_check_( true )
    , http_port_(0)
    , serv_port_(0)
    , log_level_(0)
    , last_upate_tick_(0)
{
    listen_ip_list_.clear();
}

CStreamCfg::~CStreamCfg()
{
}

int CStreamCfg::ReadCfgFile(const string& cfg_file_name)
{
    do 
    {
        CConfigHelper cfg;
        cfg.read_config_file(cfg_file_name);
        if (cfg.read_config_file(cfg_file_name)<0)
        {
            Error("read config file failed, %s", cfg_file_name.c_str());
            break;
        }

        cfg_file_name_ = cfg_file_name;
        cfg.get_value(http_port_, "setting", "http_port", 19120);
        cfg.get_value(serv_port_, "setting", "serv_port", 19020);
        cfg.get_value(log_level_, "setting", "log_level", 4);
        cfg.get_value(access_key_, "setting", "access_key", "4_odedBxmrAHiu4Y0Qp0HPG0NANCf6VAsAjWL_k9");
        cfg.get_value(secret_key_, "setting", "secret_key", "SrRuUVfDX6drVRvpyN8mv8Vcm9XnMZzlbDfvVfMe");

        {
            unsigned int value = 0;
            cfg.get_value(value, "setting", "check_token", 0);
            enable_token_check_ = value==1?true:false;
        }

        string strIpList;
        cfg.get_value(strIpList, "setting", "serv_ips", "");
        if ( !strIpList.empty() )
        {
            if (strIpList.find(",")!=std::string::npos)
            {
                CParamParser parser(",");
                parser.SetParam(strIpList.c_str());

                list<string>::iterator it=parser.m_listString.begin();
                for(; it!=parser.m_listString.end(); ++it)
                {
                    CHostInfo host(*it, serv_port_);
                    if( host.IsValid() )
                    {
                        listen_ip_list_.push_back(*it);
                    }
                }
            }
            else
            {
                CHostInfo host(strIpList, serv_port_);
                if( host.IsValid() )
                {
                    listen_ip_list_.push_back(strIpList);
                }	
            }
        }
        
        if ( listen_ip_list_.empty() )
        {
            Error("read serv_ips config failed, %s", cfg_file_name.c_str());
            break;
        }

        {
            string strTmp;
            cfg.get_value(strTmp, "setting", "record_serv", "");
            if(strTmp.empty())
            {
                strTmp = "127.0.0.1:9088";
            }
            hi_record_serv_.SetNodeString(strTmp.c_str());
        }

        {
            string strTmp;
            cfg.get_value(strTmp, "setting", "status_serv", "");
            if(strTmp.empty())
            {
                Error( "get status_serv config failed!");
                break;
            }
            hi_status_serv_.SetNodeString(strTmp.c_str());
        }

        {
            string strTmp;
            cfg.get_value(strTmp, "rtmp", "rtmp_host", "127.0.0.1");
            if(strTmp.empty())
            {
                Warn( "get rtmp_publish_host config failed!");
            }
            else
            {
                rtmp_host_ = strTmp;
            }

            cfg.get_value(rtmp_port_, "rtmp", "rtmp_port", 1935);
            cfg.get_value(rtmp_http_port_, "rtmp", "http_port", 80);
            cfg.get_value(rtmp_path_, "rtmp", "rtmp_path", "");
            cfg.get_value(rtmp_hls_path_, "rtmp", "hls_path", "");

            //rtmp_play_param=domain:pili-publish.1024.qiniu.io,test:12345
            string strRtmpPlayParamList;
            cfg.get_value(strRtmpPlayParamList, "rtmp", "rtmp_play_param", "");
            if ( !strRtmpPlayParamList.empty() )
            {
                if (strIpList.find(",")!=std::string::npos)
                {
                    CParamParser parser1(",");
                    parser1.SetParam(strIpList.c_str());

                    list<string>::iterator it=parser1.m_listString.begin();
                    for(; it!=parser1.m_listString.end(); ++it)
                    {
                        CParamParser parser2(":");
                        parser2.SetParam(strRtmpPlayParamList.c_str());

                        if( parser2.m_listString.size() == 2 )
                        {
                            list<string>::iterator it = parser2.m_listString.begin();
                            string key = *it++;
                            string value = *it;
                            rtmp_play_params_.insert(make_pair(key, value));
                        }
                    }
                }
                else
                {
                    CParamParser parser2(":");
                    parser2.SetParam(strRtmpPlayParamList.c_str());

                    if( parser2.m_listString.size() == 2 )
                    {
                        list<string>::iterator it = parser2.m_listString.begin();
                        string key = *it++;
                        string value = *it;
                        rtmp_play_params_.insert(make_pair(key, value));
                    }
                }
            }
        }

        last_upate_tick_ = get_current_tick();
        return 0;
    } while (0);
    return -1;
}

void CStreamCfg::Update()
{
    uint32 cur_tick = get_current_tick();
    if ( !last_upate_tick_ || (cur_tick-last_upate_tick_ < 60*1000) )
    {
        return;
    }

    int value = GetPrivateProfileInt("setting", "check_token", 0, cfg_file_name_.c_str());
    bool check_toke = value==1?true:false;
    if ( check_toke != enable_token_check_ )
    {
        enable_token_check_ = check_toke;
        Warn("config update, check_token=%d", check_toke);
    }
    
    int log_level = GetPrivateProfileInt("setting", "log_level", 4, cfg_file_name_.c_str());
    if ( log_level!=log_level_ )
    {
        log_level_ = log_level;
        setloglevel((Logger::LogLevel)log_level);
        Warn("config update, log_level=%d", log_level);
    }

    last_upate_tick_ = cur_tick;
}
