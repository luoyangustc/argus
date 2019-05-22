#include "StreamStatusReport.h"

static const int MAX_COUNT_PER_REQUEST			= 100;
static const int DIFF_STREAM_STATUS_TICK_LAST	= 5*1000;
static const int DIFF_STREAM_RESTART_TICK_LAST	= 10*1000;
static const int WRITE_LOG_REPORT_TIMER			= 3*60*1000;

CStreamStatusReport::CStreamStatusReport(CHostInfo hi, protocol::StsLoginReq& login_req, protocol::StsLoadReportReq& load_report)
	:login_req_(login_req)
	,load_report_(load_report)
	,hi_tracker_(hi)
    ,com_servs_agent_(NULL)
	,agent_status_(en_agent_status_init)
	,send_seq_(0)
{
	InitLoginStatus();
	SetLoadExpectedCycle();
}

CStreamStatusReport::~CStreamStatusReport(void)
{
	if(hi_tracker_.IsValid() && com_servs_agent_)
	{
		com_servs_agent_->UnadviseSink();
		com_servs_agent_->Close();
        AYClient_DestroyAYTCPClient(com_servs_agent_);
		com_servs_agent_ = NULL;
	}
}

void CStreamStatusReport::Update()
{
	std::vector<SDeviceStreamStatus> streamStatus_list;

	do
	{
		boost::lock_guard<boost::recursive_mutex> lock(lock_);
		if(stream_status_.size() < MAX_COUNT_PER_REQUEST)
		{
			stream_status_.swap(streamStatus_list);
		}
		else
		{
			streamStatus_list.reserve(MAX_COUNT_PER_REQUEST);
			::copy(stream_status_.begin(), stream_status_.begin() + MAX_COUNT_PER_REQUEST, back_inserter(streamStatus_list));
			stream_status_.erase(stream_status_.begin(), stream_status_.begin() + MAX_COUNT_PER_REQUEST);
		}

	}while(false);

	this->Update(streamStatus_list);
}

bool CStreamStatusReport::AddNotifyStatus(SDeviceStreamStatus& stream_status)
{
	boost::lock_guard<boost::recursive_mutex> lock(lock_);
	stream_status_.push_back(stream_status);
	return true;
}

ostringstream& CStreamStatusReport::DumpInfo(ostringstream& oss)
{
	oss << "{";

	oss << "\"tracker_ip\":\"";
	oss << hi_tracker_.GetNodeString();
	oss << "\"";

	oss << ",";
	oss << "\"load_expected_cycle\":\"";
	oss << load_expected_cycle_;
	oss << "\"";

	oss << ",";
	oss << "\"agent_status\":\"";
	oss << agent_status_;
	oss << "\"";

	oss << ",";
	oss << "\"last_post_load_tick\":\"";
	oss << last_report_serv_load_tick_;
	oss << "\"";

	oss << ",";
	oss << "\"last_post_device_tick\":\"";
	oss << last_report_stream_status_tick_;
	oss << "\"";

	oss << "}";
	return oss;
}

void CStreamStatusReport::SetLoadExpectedCycle(uint16 load_expected_cycle /* = 10*1000 */)
{
	load_expected_cycle_=load_expected_cycle;
}

void CStreamStatusReport::InitLoginStatus()
{
	last_report_serv_load_tick_		= 0;
	report_serv_load_ret_errcode_	= 0;
	report_serv_load_error_cnt_		= 0;

	last_report_stream_status_tick_		= 0;
	report_stream_status_ret_errcode_	= 0;
}

bool CStreamStatusReport::Restart()
{
	do 
	{
		if(!hi_tracker_.IsValid())
		{
			Error("tracker_addrs(%s) is invalid!", hi_tracker_.GetNodeString().c_str());
			break;
		}
		
		InitLoginStatus();
		SetLoadExpectedCycle();
		if( com_servs_agent_ )
		{
			com_servs_agent_->UnadviseSink();
            com_servs_agent_->Close();
            AYClient_DestroyAYTCPClient(com_servs_agent_);
            com_servs_agent_ = NULL;
		}

		com_servs_agent_ = AYClient_CreateAYTCPClient();
		if( !com_servs_agent_ )
		{
			Error("alloc tracker agent obj failed, tracker_addrs(%s).", hi_tracker_.GetNodeString().c_str());
			break;
		}
		com_servs_agent_->AdviseSink(this);

        int ret = com_servs_agent_->Connect(hi_tracker_.GetIP(), hi_tracker_.GetPort()) < 0;
        if( ret < 0 )
        {
            Error("connet to tracker_addrs(%s) failed, ret=%d.", hi_tracker_.GetNodeString().c_str(), ret);
            break;
        }
		Debug("restart tracker agent, tracker_addrs(%s).", hi_tracker_.GetNodeString().c_str());
		return true;
	} while (false);
	return false;
}

void CStreamStatusReport::Update(std::vector<SDeviceStreamStatus>& stream_status)
{
	uint32 current_tick_count = get_current_tick();
	switch(agent_status_)
	{
	case en_agent_status_init:
	case en_agent_status_error:
		{
			if((current_tick_count - last_status_chg_tick_) > DIFF_STREAM_RESTART_TICK_LAST)
			{
				agent_status_ = en_agent_status_connecting;
				if( !Restart() )
				{
					agent_status_ = en_agent_status_error;
				}
				last_status_chg_tick_ = current_tick_count;
			}
			break;
		}
	case en_agent_status_connected:
		{
			ReportLoginReq();
			break;
		}
	case en_agent_status_connecting:
	case en_agent_status_logining:
		{
			if((current_tick_count - last_status_chg_tick_) > DIFF_STREAM_RESTART_TICK_LAST)
			{
				Error("agent login timeout, tracker(%s) status(%u)!", hi_tracker_.GetNodeString().c_str(), (int)agent_status_);
				agent_status_ = en_agent_status_error;
			}
			break;
		}
	case en_agent_status_logined:
		{
			if( current_tick_count - last_report_serv_load_tick_ > GetLoadExpectedCycle() )
			{
				if(report_serv_load_ret_errcode_ != 0)
				{
					report_serv_load_error_cnt_ += 1;
					if(report_serv_load_error_cnt_ > 3)
					{
						agent_status_ = en_agent_status_error;
						Error("report server, load failed tracker(%s)!", hi_tracker_.GetNodeString().c_str());
						break;
					}
				}
				else
				{
					report_serv_load_error_cnt_ = 0;
				}

				ReportServLoad();
			}

			if(report_stream_status_ret_errcode_ !=0)
			{
				if(current_tick_count - last_report_stream_status_tick_ > DIFF_STREAM_STATUS_TICK_LAST)
				{
					agent_status_ = en_agent_status_error;
					Error("report stream status, failed tracker(%s)!", hi_tracker_.GetNodeString().c_str());
				}
				break;
			}
			ReportStreamStatus(stream_status);
			break;
		}
	default:
		{
			Error("agent is error,tracker(%s) agent_status_(%u)!", hi_tracker_.GetNodeString().c_str(), (uint16)agent_status_);
			agent_status_ = en_agent_status_error;
		}
		break;
	}
}

void CStreamStatusReport::ReportLoginReq()
{
    do 
    {
        char buffer[1024];
        CDataStream sendds((unsigned char *)buffer, 1024);

        MsgHeader msg_header;
        msg_header.msg_id = MSG_ID_STS_LOGIN;
        msg_header.msg_type = MSG_TYPE_REQ;
        msg_header.msg_seq = ++send_seq_;
        sendds << msg_header;

        sendds << login_req_;
        *((WORD*)sendds.getbuffer()) = sendds.size();

        int ret = com_servs_agent_->Send((const unsigned char*)sendds.getbuffer(),sendds.size());
        if(ret < 0)
        {
            agent_status_ = en_agent_status_error;
            last_status_chg_tick_ = get_current_tick();
            Error("send login request msg failed, msgid(%u) hostinfo(%s) serv_port(%u) http_port(%u), ret=%d", 
                (uint32)MSG_ID_STS_LOGIN, 
                hi_tracker_.GetNodeString().c_str(), 
                login_req_.serv_port, 
                login_req_.http_port, ret );
            break;
        }
        agent_status_ = en_agent_status_logining;
        last_status_chg_tick_ = get_current_tick();
        Debug("send login request msg, msgid(%u) hostinfo(%s) serv_port(%u) http_port(%u)", 
            (uint32)MSG_ID_STS_LOGIN, 
            hi_tracker_.GetNodeString().c_str(), 
            login_req_.serv_port, 
            login_req_.http_port );
    } while (0);
    
}

void CStreamStatusReport::ReportServLoad()
{
    do 
    {
        report_serv_load_ret_errcode_ = -1;
        last_report_serv_load_tick_ = get_current_tick();

        char buffer[1024];
        CDataStream sendds((unsigned char *)buffer, 1024);

        MsgHeader msg_header;
        msg_header.msg_id	= MSG_ID_STS_LOAD_REPORT;
        msg_header.msg_type = MSG_TYPE_REQ;
        msg_header.msg_seq	= ++send_seq_;
        sendds << msg_header;

        sendds << load_report_;
        *((WORD*)sendds.getbuffer()) = sendds.size();

        static tick_t last_write_log_tick = 0;
        if (get_current_tick()-last_write_log_tick>WRITE_LOG_REPORT_TIMER)
        {
            Debug("report stream server load, msgid(%u) hostinfo(%s) last_tick(%u) msg_size(%u)", (uint32)MSG_ID_STS_LOAD_REPORT, hi_tracker_.GetNodeString().c_str(), (uint32)last_report_serv_load_tick_, sendds.size());
            last_write_log_tick = get_current_tick();
        }

        int ret = com_servs_agent_->Send((const unsigned char*)sendds.getbuffer(),sendds.size());
        if(ret < 0)
        {
            agent_status_ = en_agent_status_error;
            last_status_chg_tick_ = get_current_tick();
            Error("send load report request msg failed, hostinfo(%s) serv_port(%u) http_port(%u), ret=%d",
                hi_tracker_.GetNodeString().c_str(), 
                login_req_.serv_port, 
                login_req_.http_port, ret );
            break;
        }
    } while (0);
}

void CStreamStatusReport::ReportStreamStatus(std::vector<SDeviceStreamStatus>& stream_status_list)
{
    do 
    {
        if (stream_status_list.size()<=0)
        {
            return;
        }
        report_stream_status_ret_errcode_	= -1;
        last_report_stream_status_tick_		= get_current_tick();

        uint32 i_device_num = stream_status_list.size();	
        uint32 i_msg_buf_size = sizeof(StsStreamStatusReportReq) + sizeof(SDeviceStreamStatus) * i_device_num * 2; 

        boost::shared_array<uint8> send_buff(new uint8[i_msg_buf_size]);
        CDataStream sendds(send_buff.get(), i_msg_buf_size);

        MsgHeader msg_header;
        msg_header.msg_id	= MSG_ID_STS_STREAM_STATUS_REPORT;
        msg_header.msg_type = MSG_TYPE_REQ;
        msg_header.msg_seq	= ++send_seq_;
        sendds << msg_header;

        StsStreamStatusReportReq stream_report;
        stream_report.mask			= 0x01;
        stream_report.device_num	= i_device_num;
        stream_report.devices.assign(stream_status_list.begin(), stream_status_list.end());
        sendds << stream_report;
        *((WORD*)sendds.getbuffer()) = sendds.size();

        int ret = com_servs_agent_->Send((const unsigned char*)sendds.getbuffer(),sendds.size());
        if(ret < 0)
        {
            agent_status_ = en_agent_status_error;
            last_status_chg_tick_ = get_current_tick();
            Error("send device status report request msg failed, msgid(%u) hostinfo:%s device_num:%u msg_size:%u, ret=%d",
                (uint32)MSG_ID_STS_STREAM_STATUS_REPORT, 
                hi_tracker_.GetNodeString().c_str(), 
                i_device_num, 
                sendds.size(), ret );
            break;
        }

        Debug("report device stream status, msgid(%u) hostinfo:%s device_num:%u msg_size:%u", 
            (uint32)MSG_ID_STS_STREAM_STATUS_REPORT, 
            hi_tracker_.GetNodeString().c_str(), 
            i_device_num, 
            sendds.size());
    } while (0);
    
	
}

bool CStreamStatusReport::OnServLoginResponse(uint8 * pData,uint32 data_len)
{
	do 
	{
		CDataStream recvds(pData,data_len);	
		MsgHeader msg_header;
		recvds >> msg_header;
		StsLoginResp res;
		recvds >> res;
		if (!recvds.good_bit())
		{
			Error("response login recv invalid msg, tracker(%s)!", hi_tracker_.GetNodeString().c_str());
			break;
		}

		if( res.resp_code == 0 )
		{
			InitLoginStatus();
			agent_status_ = en_agent_status_logined;
		}
		else
		{
			agent_status_ = en_agent_status_error;
			Error("response login, tracker(%s) error_code(%d)!", hi_tracker_.GetNodeString().c_str(), res.resp_code);
			break;
		}

		Debug("response login, tracker(%s) load_expected_cycle(%u)!", hi_tracker_.GetNodeString().c_str(), (uint32)res.load_expected_cycle);
		SetLoadExpectedCycle(res.load_expected_cycle*1000);

		return true;
	} while(false);
	return false;
}

bool CStreamStatusReport::OnServLoadResponse(uint8 * pData,uint32 data_len)
{
	do 
	{
		CDataStream recvds(pData,data_len);		

		MsgHeader msg_header;
		recvds >> msg_header;			
		
		StsLoadReportResp res;
		recvds >> res;
		
		if (!recvds.good_bit())
		{
			Error("response load recv invalid msg, tracker(%s)!", hi_tracker_.GetNodeString().c_str());
			break;
		}

		//Debug("response load, tracker(%s) load_expected_cycle(%u)!", hi_tracker_.GetNodeString().c_str(), (uint32)res.load_expected_cycle);
		SetLoadExpectedCycle(res.load_expected_cycle*1000);
		report_serv_load_ret_errcode_ = 0;

		return true;
	} while(false);
	return false;
}

bool CStreamStatusReport::OnServStatusResponse(uint8 * pData,uint32 data_len)
{
	do 
	{
		CDataStream recvds(pData,data_len);	

		MsgHeader msg_header;
		recvds >> msg_header;	

		StsStreamStatusReportResp res;
		recvds >> res;
		
		if (!recvds.good_bit())
		{
			Error("response status recv invalid msg, tracker(%s)!", hi_tracker_.GetNodeString().c_str());
			break;
		}

		report_stream_status_ret_errcode_ = 0;
		Debug("response status, tracker(%s) code(%u)!", hi_tracker_.GetNodeString().c_str(), (uint32)res.resp_code);
		return true;
	} while(false);
	return false;
}

int CStreamStatusReport::OnTCPConnected(uint32 ip, uint16 port)
{
	Debug("tracker(%s), current_status:(%u)", hi_tracker_.GetNodeString().c_str(),(int)agent_status_);
	if(agent_status_ != en_agent_status_connecting)
	{
		agent_status_ = en_agent_status_error;
	}
	else
	{
		agent_status_ = en_agent_status_connected;
	}
    return 0;
}

int CStreamStatusReport::OnTCPConnectFailed(uint32 ip, uint16 port)
{
	agent_status_ = en_agent_status_error;
	Warn("connect failed, tracker(%s)!", hi_tracker_.GetNodeString().c_str());
    return 0;
}

int	CStreamStatusReport::OnTCPMessage(uint32 ip, uint16 port, uint8* data, uint32 data_len)
{
    CDataStream recvds(data, data_len);
    protocol::MSG_HEADER header;
    recvds >> header;

    CHostInfo hi_remote(ip, port);
    // Debug("from(%s),msg_id=0x%x, data_len:%u", hi_remote.GetNodeString().c_str(), header.msg_id, data_len);

	bool b_code = false;
	switch(header.msg_id)
	{
	case MSG_ID_STS_LOGIN:
		{
			b_code = OnServLoginResponse(data,data_len);
			break;		
		}
	case MSG_ID_STS_LOAD_REPORT:
		{
			b_code = OnServLoadResponse(data,data_len);
			break;		
		}
	case MSG_ID_STS_STREAM_STATUS_REPORT:
		{
			b_code = OnServStatusResponse(data,data_len);
			break;
		}
	default:
		{
			Error("recv unknown message, msg_id(%u) tracker_addrs(%s)!", header.msg_id, hi_tracker_.GetNodeString().c_str());
			break;
		}
	}
	return b_code?data_len:-1;
}

int CStreamStatusReport::OnTCPClose(uint32 ip, uint16 port)
{
	agent_status_ = en_agent_status_error;
	Warn("connect closed, tracker(%s)!", hi_tracker_.GetNodeString().c_str());
    return 0;
}
