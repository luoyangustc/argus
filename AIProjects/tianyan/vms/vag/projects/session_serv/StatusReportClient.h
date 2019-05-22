#ifndef __STATUS_REPORT_CLIENT_H__
#define __STATUS_REPORT_CLIENT_H__

#include <time.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/atomic.hpp>
#include "base/include/typedef_win.h"
#include "base/include/HostInfo.h"
#include "protocol/include/protocol_status.h"
#include "netlib_framework/include/IClientSocket.h"
#include "netlib_framework/include/IClientSocketSink.h"

using namespace std;
using namespace protocol;

 class CStatusReportClient : public ITCPClientSink
 {
 public:
    CStatusReportClient(CHostInfo hi,protocol::StsLoginReq& login_req,protocol::StsLoadReportReq& load_report);
    virtual ~CStatusReportClient(void);
    bool PushNotifyStatus(SDeviceSessionStatus& device_status);
    void Update();
public:
    virtual int OnTCPMessage(uint32 ip, uint16 port, uint8* data, uint32 data_len) ;
    virtual int OnTCPConnected(uint32 ip,uint16 port);
    virtual int OnTCPConnectFailed(uint32 ip, uint16 port);
    virtual int OnTCPClose(uint32 ip, uint16 port);
    void Update(vector<SDeviceSessionStatus>& deviceStatus);
    ostringstream& DumpInfo(ostringstream& oss);
private:
    bool ON_ServLoginResponse(uint8 * pData,uint32 data_len);
    bool ON_ServLoadResponse(uint8 * pData,uint32 data_len);
    bool ON_ServStatusResponse(uint8 * pData,uint32 data_len);
    void ReportDeviceStatus(vector<SDeviceSessionStatus>& deviceStatus);
    void ReportServLoad();
    bool Restart();
    void AgentStatusClear();
private:
    boost::recursive_mutex lock_;
    protocol::StsLoginReq& login_req_;
    protocol::StsLoadReportReq& load_report_;
    CHostInfo hi_tracker_;
    ITCPClient* tracker_agent_;
    uint16  load_expected_cycle_;  //负载期望周期，单位为秒
 private:
    enum EnAgentStatus
    {
        en_agent_status_init = 0,
        en_agent_status_connecting,
        en_agent_status_connected,
        en_agent_status_logining,
        en_agent_status_logined,
        en_agent_status_error
    };
    boost::atomic_uint agent_status_;
    boost::atomic_uint last_status_chg_tick_;
    boost::atomic_uint last_report_serv_load_tick_;
    boost::atomic_uint last_report_dev_status_tick_;
    boost::atomic_int report_serv_load_ret_errcode_;
    boost::atomic_int report_dev_status_ret_errcode_;
    boost::atomic_uint report_serv_load_error_cnt_;
private:
    boost::atomic_uint msg_seq_;
    vector<SDeviceSessionStatus> deviceStatus_;
    static const int MAX_COUNT_PER_REQUEST = 50;
};

typedef boost::shared_ptr<CStatusReportClient> CStatusReportClient_ptr;

#endif 

