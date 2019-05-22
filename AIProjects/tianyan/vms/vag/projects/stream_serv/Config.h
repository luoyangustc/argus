#ifndef __CONFIG_H__
#define __CONFIG_H__

#include "CommonInc.h"

class CStreamCfg
{
public:
    CStreamCfg();
    ~CStreamCfg();
    int ReadCfgFile(const string& cfg_file_name);
    void Update();
public:
    uint32 GetHttpPort(){return http_port_;}
    uint32 GetServPort(){return serv_port_;}
    CHostInfo GetRecordServ(){return hi_record_serv_;}
    CHostInfo GetStatusServ(){return hi_status_serv_;}
    const vector<string>& GetListenIpList(){return listen_ip_list_;}
    string GetAccessKey(){return access_key_;}
    string GetSecretKey(){return secret_key_;}
public:
    string GetRtmpHost(){return rtmp_host_;}
    uint32 GetRtmpPort(){return rtmp_port_;}
    uint32 GetRtmpHttpPort(){return rtmp_http_port_;}
    string GetRtmpPath(){return rtmp_path_;}
    string GetRtmpHlsPath(){return rtmp_hls_path_;}
    const map<string, string>& GetRtmpPlayParams(){ return rtmp_play_params_; }
public:
    bool IsTokenCheck(){return enable_token_check_;}
    uint32 GetLogLevel(){return log_level_;}
private:
    boost::recursive_mutex lock_;
    string cfg_file_name_;
    uint32 last_upate_tick_;

    //必须重启生效的配置项
    uint32 http_port_;
    uint32 serv_port_;
    CHostInfo hi_record_serv_;
    CHostInfo hi_status_serv_;
    vector<string> listen_ip_list_;
    string access_key_;
    string secret_key_;

    string rtmp_host_;
    uint32 rtmp_port_;
    uint32 rtmp_http_port_;
    string rtmp_path_;
    string rtmp_hls_path_;
    map<string, string> rtmp_play_params_;

    
    //可动态更新的配置项
    bool enable_token_check_;
    uint32 log_level_;
};

typedef boost::shared_ptr<CStreamCfg> CStreamCfg_ptr;

#endif
