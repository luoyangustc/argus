#ifndef _STREAMSTATUSREPORT_H
#define _STREAMSTATUSREPORT_H
#include "CommonInc.h"

//agent login status code
enum EnAgentStatus
{
	en_agent_status_init = 0,
	en_agent_status_connecting,
	en_agent_status_connected,
	en_agent_status_logining,
	en_agent_status_logined,
	en_agent_status_error
};
class CStreamStatusReport : public ITCPClientSink
{
public:
	CStreamStatusReport(CHostInfo hi, protocol::StsLoginReq& login_req, protocol::StsLoadReportReq& load_report);
	~CStreamStatusReport(void);

	bool	AddNotifyStatus(SDeviceStreamStatus& stream_status);
	void	Update();
	ostringstream& DumpInfo(ostringstream& oss);
	void    SetLoadExpectedCycle(uint16 load_expected_cycle = 10*1000);

public:
    virtual int OnTCPConnected(uint32 ip, uint16 port);
    virtual int OnTCPConnectFailed(uint32 ip, uint16 port);
    virtual int OnTCPClose(uint32 ip, uint16 port);
    virtual int OnTCPMessage(uint32 ip, uint16 port, uint8* data, uint32 data_len);

private:
	void	InitLoginStatus();
	bool    Restart();
	void	Update(std::vector<SDeviceStreamStatus>& stream_status);
	void	ReportStreamStatus(std::vector<SDeviceStreamStatus>& stream_status_list);
	void	ReportServLoad();
	void	ReportLoginReq();

	bool	OnServLoginResponse(uint8 * pData, uint32 data_len);
	bool	OnServLoadResponse(uint8 * pData, uint32 data_len);
	bool	OnServStatusResponse(uint8 * pData, uint32 data_len);
	uint32	GetLoadExpectedCycle(){return load_expected_cycle_;}
private:
	CHostInfo					hi_tracker_;
	boost::recursive_mutex		lock_;
	protocol::StsLoginReq&		login_req_;
	protocol::StsLoadReportReq&	load_report_;
	uint16						load_expected_cycle_;
	ITCPClient*                 com_servs_agent_;

	std::vector<SDeviceStreamStatus> stream_status_;
	boost::atomic_uint	send_seq_;
	//status code;
	boost::atomic_uint	agent_status_;
	boost::atomic_uint	last_status_chg_tick_;

	boost::atomic_uint	last_report_stream_status_tick_;
	boost::atomic_int	report_stream_status_ret_errcode_;

	boost::atomic_uint	last_report_serv_load_tick_;
	boost::atomic_int	report_serv_load_ret_errcode_;
	boost::atomic_uint	report_serv_load_error_cnt_;
};

typedef boost::shared_ptr<CStreamStatusReport> CStreamStatusReport_ptr;

#endif  //_STREAMSTATUSREPORT_H