#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <string>
#include <vector>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/shared_ptr.hpp>
#include "HostInfo.h"
#include "typedefine.h"

using namespace std;

class CServerCfg
{
public:
    CServerCfg();
    ~CServerCfg();
    int ReadCfgFile(const string& cfg_file_name);
    void Update();
public:
	string GetServIp();
    uint32 GetHttpPort(){return http_port_;}
    uint32 GetServPort(){return serv_port_;}
    const vector<string>& GetListenIpList(){return listen_ip_list_;}
    string GetAccessKey() {return access_key_;}
    string GetSecretKey(){return secret_key_;}
    
    string& GetGlsb() {return glsb_host_;}
    CHostInfo GetStatusServ(){return hi_status_serv_;}

    bool IsTokenCheck(){return enable_token_check_;}
    uint32 GetLogLevel(){return log_level_;}

    string& GetSnapPicSavePath(){return pic_save_path_;}
    string& GetSnapPicUrl(){return pic_url_;}
private:
    boost::recursive_mutex lock_;
    string cfg_file_name_;
    uint32 last_upate_tick_;

    //必须重启生效的配置项
    uint32 http_port_;
    uint32 serv_port_;
    vector<string> listen_ip_list_;

    string glsb_host_;
    CHostInfo hi_status_serv_;

    string access_key_;
    string secret_key_;
    
    //可动态更新的配置项
    bool enable_token_check_;
    uint32 log_level_;

    //snap
    string pic_save_path_;
    string pic_url_;
};

typedef boost::shared_ptr<CServerCfg> CServerCfg_ptr;

#endif
