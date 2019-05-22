#ifndef C5_UTIL_SERV_SERVERLOGIC_H
#define C5_UTIL_SERV_SERVERLOGIC_H

#include <signal.h>

#include <set>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/scoped_array.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/shared_array.hpp>

#include "IServerLogical.h"

#include "typedefine.h"
#include "common_thread_base.h"
#include "tick.h"
#include "ConfigHelper.h"
#include "logging_posix.h"

class CSysMonitorThread : public CCommon_Thread_Base
{
public:
    CSysMonitorThread() {
        last_active_tick_ = get_current_tick();
    }
    ~CSysMonitorThread() {
        Stop();
    }

    void UpdateActiveTick() {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        last_active_tick_ = get_current_tick();
    }

    virtual bool Run() {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        tick_t current_tick = get_current_tick();
        if ( current_tick - last_active_tick_ > 30*1000 )
        {
            Fatal("[CCheckActiveThread] Exit current_tick(%lld), last_active_tick(%lld)",
                current_tick, last_active_tick_);

            std::string ofn = CConfigHelper::get_default_config_dir() + "/deadlock_stack.txt";

            time_t now = time(NULL);
            struct tm ntm;
            localtime_r(&now, &ntm);
            char time_buf[256] = {0};
            sprintf(time_buf, "%d-%02d-%02d %02d:%02d:%02d", 
                ntm.tm_year + 1900, ntm.tm_mon + 1, ntm.tm_mday,
                ntm.tm_hour, ntm.tm_min, ntm.tm_sec);

            char cmd[512];
            snprintf(cmd,sizeof(cmd),"echo -e \"----------------%s deadlock stack:\r\n\" >> %s", 
                time_buf, ofn.c_str());
            std::system(cmd);

            snprintf(cmd,sizeof(cmd),"pstack %d >> %s", getpid(), ofn.c_str());
            std::system(cmd);

            snprintf(cmd,sizeof(cmd),"echo -e \"\r\n\r\n\r\n\r\n\r\n\r\n\" >> %s", ofn.c_str());
            std::system(cmd);

            exit(0);
            return false;
        }

        return true;
    }

private:
    boost::recursive_mutex lock_;
    tick_t last_active_tick_;
};

typedef boost::shared_ptr<CSysMonitorThread> CSysMonitorThread_ptr;

class CServerLogical : public IServerLogical
{
public:
    enum { kErrorNone=0, kInvalidParameter=2001, kServerInternalError}; 
    CServerLogical();
    virtual ~CServerLogical();

    virtual bool Start(uint16 http_port,uint16 serv_port);  
    virtual void Stop();

    virtual int32 OnHttpClientRequest(ITCPSessionSendSink*sink,
        CHostInfo& hiRemote,
        SHttpRequestPara_ptr pReq,SHttpResponsePara_ptr pRes);
    virtual int32 OnUDPMessage(CHostInfo& hiRemote, 
        CDataStream& recvds, CDataStream& sendds, IN int thread_index) { return -1; }
    virtual int32 OnTCPMessage(ITCPSessionSendSink*sink,
        CHostInfo& hiRemote,
        uint32 msg_id,
        CDataStream& recvds,
        CDataStream& sendds) { return -1; }
    virtual int32 OnTCPAccepted(ITCPSessionSendSink*sink,
        CHostInfo& hiRemote,CDataStream& sendds);
    virtual int32 OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& hiRemote);
    virtual void Update();
    virtual void DoIdleTask();

    static CServerLogical* GetLogical() { return &logic_; }

private:
    typedef std::map<std::string,std::string> HttpHeaderMap;
    typedef std::map<std::string,std::string>::iterator HttpHeaderIterator; 

    int HandleHttpRequest(const std::string& page,
        HttpHeaderMap& params,
        SHttpResponsePara_ptr& httpResp);  
    int DefaultHandle(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int GenerateDeviceId(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int CheckDeviceId(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int GenerateDeviceToken(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int DecryptDeviceToken(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int GenerateUserToken(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int DecryptUserToken(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int EncryptString(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int DecryptString(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int HttpNotFound(SHttpResponsePara_ptr& pRes);  
    void MakeHttpResponse(const std::string& content, SHttpResponsePara_ptr& pRes);
    void NotHandleOk(SHttpResponsePara_ptr& pRes);   

    int GetAccessToken(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int GetAccessServer(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
    int StreamRRS(HttpHeaderMap& params, SHttpResponsePara_ptr& pRes);
private:
    bool has_private_ip_;
    tick_t start_tick_;
    std::list<uint32> local_ips_;
    uint32 local_first_ip_;
    CSysMonitorThread_ptr pSysMonitorThread_;

    static CServerLogical logic_;   
};

#endif  // C5_UTIL_SERV_SERVERLOGIC_H