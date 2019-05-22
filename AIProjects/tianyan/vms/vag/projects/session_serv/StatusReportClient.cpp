#include "StatusReportClient.h"
#include "base/include/tick.h"
#include "base/include/logging_posix.h"
#include "DeviceMgr.h"
#include "ServerLogical.h"
#include "netlib_framework/include/AYClientApi.h"

CStatusReportClient::CStatusReportClient(CHostInfo hi,protocol::StsLoginReq& login_req, protocol::StsLoadReportReq& load_report)
    :login_req_(login_req)
    ,load_report_(load_report)
    ,tracker_agent_(NULL)
    ,hi_tracker_(hi)
    ,msg_seq_(0)
{
    printf( "CStatusReportClient::CStatusReportClient-->login_req_[serv_port=%u, http_port=%u]\n", 
        login_req_.serv_port, login_req_.http_port );
    load_expected_cycle_ = 10;
    agent_status_ = en_agent_status_init;
    AgentStatusClear();
}

CStatusReportClient::~CStatusReportClient(void)
{
    if(tracker_agent_)
    {
        tracker_agent_->UnadviseSink();
        tracker_agent_->Close();
        AYClient_DestroyAYTCPClient(tracker_agent_);
        tracker_agent_ = NULL;
    }
}

void CStatusReportClient::AgentStatusClear()
{
    report_serv_load_ret_errcode_ = 0;
    report_dev_status_ret_errcode_ = 0;
    report_serv_load_error_cnt_ = 0;
    last_report_serv_load_tick_ = 0;
    last_report_dev_status_tick_ = 0;
}

bool CStatusReportClient::ON_ServLoginResponse(uint8 * pData,uint32 data_len)
{
    do 
    {
        CDataStream recvds(pData,data_len);
        StsLoginResp res;
        recvds >> res;
        if (!recvds.good_bit())
        {
            Error("tracker(%s), recv invalid msg!", 
                hi_tracker_.GetNodeString().c_str());
            break;
        }

        if( res.resp_code == 0 )
        {
            AgentStatusClear();
            agent_status_ = en_agent_status_logined;
        }
        else
        {
            agent_status_ = en_agent_status_error;
            Error("tracker(%s),error_code=%d!", 
                hi_tracker_.GetNodeString().c_str(), res.resp_code);
            break;
        }

        load_expected_cycle_ = res.load_expected_cycle;

        CDeviceMgr_ptr pDevMgr = CServerLogical::GetLogical()->GetDeviceMgr();
        vector<SDeviceSessionStatus> deviceStatus;
        pDevMgr->GenDeviceStatus(deviceStatus);

        if (!deviceStatus.empty())
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            ::copy(deviceStatus.begin(), deviceStatus.end(), back_inserter(deviceStatus_));
        }

        Debug("tracker(%s), load_expected_cycle(%u)!", 
            hi_tracker_.GetNodeString().c_str(), 
            (uint32)load_expected_cycle_);

		return true;
	} while(false);

	return false;
}

bool CStatusReportClient::ON_ServLoadResponse(uint8 * pData,uint32 data_len)
{
	do 
	{
		CDataStream recvds(pData,data_len);
		StsLoadReportResp res;
		recvds >> res;
		if (!recvds.good_bit())
		{
            Error("tracker(%s), recv invalid msg!", hi_tracker_.GetNodeString().c_str());
			break;
		}

        load_expected_cycle_ = res.load_expected_cycle;
        report_serv_load_ret_errcode_ = 0;
		return true;

	} while(false);

	return false;
}

bool CStatusReportClient::ON_ServStatusResponse(uint8 * pData,uint32 data_len)
{
    do 
    {
        CDataStream recvds(pData,data_len);
        StsSessionStatusReportResp res;
        recvds >> res;
        if (!recvds.good_bit())
        {
            Error("tracker(%s), recv invalid msg!", 
				hi_tracker_.GetNodeString().c_str());
            break;
        }

        report_dev_status_ret_errcode_ = 0;

        return true;
    } while(false);

    return false;

}

int CStatusReportClient::OnTCPConnected(uint32 ip,uint16 port)
{
    Debug("tracker(%s), current_status:(%u)",
        hi_tracker_.GetNodeString().c_str(),(int)agent_status_);

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

int CStatusReportClient::OnTCPConnectFailed(uint32 ip, uint16 port)
{
    agent_status_ = en_agent_status_error;
    Debug("tracker(%s), connect failed!", hi_tracker_.GetNodeString().c_str());
    
}

int CStatusReportClient::OnTCPClose(uint32 ip, uint16 port)
{
    agent_status_ = en_agent_status_error;
    Debug("tracker(%s), connect closed!", hi_tracker_.GetNodeString().c_str());
}

int CStatusReportClient::OnTCPMessage(uint32 ip, uint16 port, uint8* data, uint32 data_len)
{
    CDataStream recvds(data, data_len);
    protocol::MSG_HEADER header;
    recvds >> header;

    CHostInfo hi_remote(ip, port);
    //Debug("from(%s),msg_id=0x%x, data_len:%u", hi_remote.GetNodeString().c_str(), header.msg_id, data_len);

    uint8* body = (uint8*)recvds.getcurrent_pos();
    uint32 body_len = recvds.leavedata();

    switch(header.msg_id)
    {
    case MSG_ID_STS_LOGIN:
        {
            if( ON_ServLoginResponse(body,body_len) )
            {
                return data_len;
            }			
        }
        break;
    case MSG_ID_STS_LOAD_REPORT:
        {
            if( ON_ServLoadResponse(body,body_len) )
            {
                return data_len;
            }			
        }
        break;
    case MSG_ID_STS_SESSION_STATUS_REPORT:
        {
            if( ON_ServStatusResponse(body,body_len) )
            {
                return data_len;
            }
        }
        break;
    default:
        {
            Error("tracker_addrs(%s), recv unknown message!", hi_tracker_.GetNodeString().c_str());
        }
        break;
    }

    return -1;
}

bool CStatusReportClient::Restart()
{
    do 
    {
        if(!hi_tracker_.IsValid())
        {
            Error("tracker_addrs(%s) is invalid!", hi_tracker_.GetNodeString().c_str());
            break;
        }

        //public_ip_ = 0;
        //type_ = 0;
        //need_report_dev_status_flag_ = false;
        load_expected_cycle_ = 10;

        AgentStatusClear();

        if( tracker_agent_ )
        {
            tracker_agent_->UnadviseSink();
            tracker_agent_->Close();
            AYClient_DestroyAYTCPClient(tracker_agent_);
            tracker_agent_ = NULL;
        }

        tracker_agent_ = AYClient_CreateAYTCPClient();
        if( !tracker_agent_ )
        {
            Error("alloc tracker agent obj failed, tracker_addrs(%s).", hi_tracker_.GetNodeString().c_str());
            break;
        }

        tracker_agent_->AdviseSink(this);
        int ret = tracker_agent_->Connect(hi_tracker_.GetIP(), hi_tracker_.GetPort());

        Debug("restart tracker agent, tracker_addrs(%s), ret=%d.", hi_tracker_.GetNodeString().c_str(), ret);

        return true;

    } while (0);
    
    return false;
}

bool CStatusReportClient::PushNotifyStatus(SDeviceSessionStatus& device_status)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    deviceStatus_.push_back(device_status);
    return true;
}

void CStatusReportClient::Update()
{
    vector<SDeviceSessionStatus> deviceStatus_list;

    if (agent_status_ == en_agent_status_logined)
    {
        do
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            if(deviceStatus_.size() < MAX_COUNT_PER_REQUEST)
            {
                deviceStatus_.swap(deviceStatus_list);
            }
            else
            {
                deviceStatus_list.reserve(MAX_COUNT_PER_REQUEST);
                ::copy(deviceStatus_.begin(), deviceStatus_.begin() + MAX_COUNT_PER_REQUEST, back_inserter(deviceStatus_list));
                deviceStatus_.erase(deviceStatus_.begin(), deviceStatus_.begin() + MAX_COUNT_PER_REQUEST);
            }

        }while(false);
    }

    if (!deviceStatus_list.empty())
    {
        char buf[4096];
        int  buf_bytes = 0;
        for(vector<SDeviceSessionStatus>::iterator it=deviceStatus_list.begin(); it!=deviceStatus_list.end(); it++)
        {
            buf_bytes += snprintf(buf+buf_bytes,sizeof(buf)-buf_bytes,"%s,", it->did.c_str());
        }
        buf_bytes = buf_bytes<sizeof(buf) ? buf_bytes : sizeof(buf);
        buf[buf_bytes-1] = '\0';
        Debug("begin to report device status, device id list: %s", buf);
    }

    this->Update(deviceStatus_list);
}

void CStatusReportClient::Update(vector<SDeviceSessionStatus>& deviceStatus)
{
    bool hasReport = false;
    int ret = 0;
    DWORD current_tick_count = get_current_tick();
    switch(agent_status_)
    {
    case en_agent_status_init:
    case en_agent_status_error:
        {
            if((current_tick_count - last_status_chg_tick_) > 10000)
            {
                agent_status_ = en_agent_status_connecting;
                if( !Restart() )
                {
                    agent_status_ = en_agent_status_error;
                }
                last_status_chg_tick_ = current_tick_count;
            }
        }
        break;
    case en_agent_status_connected:
        {
            agent_status_ = en_agent_status_logining;
            last_status_chg_tick_ = current_tick_count;

            BYTE buffer[1024];
            CDataStream sendds((unsigned char *)buffer, 1024);
            protocol::MSG_HEADER header;
            header.msg_id = MSG_ID_STS_LOGIN;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++msg_seq_;
            sendds << header;
            sendds << login_req_;
            *(uint16*)sendds.getbuffer() = sendds.size();
            if( ret = tracker_agent_->Send((const unsigned char*)sendds.getbuffer(),sendds.size()) < 0 )
            {
                agent_status_ = en_agent_status_error;
            }
            else
            {
                agent_status_ = en_agent_status_logining;
            }
            last_status_chg_tick_ = current_tick_count;

            Debug("send login request msg to:%s, serv_port:%u, http_port:%u, ret=%d", 
                hi_tracker_.GetNodeString().c_str(), 
                login_req_.serv_port, 
                login_req_.http_port, ret );
        }
        break;
    case en_agent_status_connecting:
    case en_agent_status_logining:
        {
            if((current_tick_count - last_status_chg_tick_) > 10000)
            {
                Error("timeout, tracker(%s), status(%u)!", hi_tracker_.GetNodeString().c_str(), (int)agent_status_);
                agent_status_ = en_agent_status_error;
            }
        }
        break;
    case en_agent_status_logined:
        {
            //① 负载上报处理
            if( current_tick_count - last_report_serv_load_tick_ > load_expected_cycle_ * 1000 )
            {
                //检查上次上报是否出错
                if(report_serv_load_ret_errcode_ != 0)
                {
                    report_serv_load_error_cnt_ += 1;
                    if(report_serv_load_error_cnt_ > 3)
                    {
                        agent_status_ = en_agent_status_error;
                        Error("tracker(%s), report server load failed!", hi_tracker_.GetNodeString().c_str());
                        break;
                    }
                }
                else
                {
                    report_serv_load_error_cnt_ = 0;
                }

                //上报负载
                ReportServLoad();
            }

            //② 设备状态上报
            //检查上次上报是否出错
            if(report_dev_status_ret_errcode_ !=0)
            {
                if(current_tick_count - last_report_dev_status_tick_ > 5000)
                {
                    agent_status_ = en_agent_status_error;
                    Error("tracker(%s), report device status failed!", hi_tracker_.GetNodeString().c_str());
                }
                break;
            }

            //上报设备状态
            ReportDeviceStatus(deviceStatus);
            hasReport = true;
        }
        break;
    default:
        {
            Error("tracker(%s), agent_status_(%u) is error!", hi_tracker_.GetNodeString().c_str(), (int)agent_status_);

            agent_status_ = en_agent_status_error;
        }
        break;
    }

    if (!hasReport)
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        for(vector<SDeviceSessionStatus>::iterator it=deviceStatus.begin(); it!=deviceStatus.end(); it++)
        {
            deviceStatus_.push_back(*it);
        }
    }
}

void CStatusReportClient::ReportDeviceStatus(vector<SDeviceSessionStatus>& deviceStatus)
{
    if (deviceStatus.empty())
    {
        return;
    }

    report_dev_status_ret_errcode_ = -1;
    last_report_dev_status_tick_ = get_current_tick();

    uint32 deviceNum = deviceStatus.size();	
	uint32 msg_buf_size = sizeof(StsSessionStatusReportReq) + sizeof(SDeviceSessionStatus) * deviceNum * 100; 

	StsSessionStatusReportReq device_report;
	device_report.mask = 0x01;
    device_report.device_num = deviceNum;
    device_report.devices.assign(deviceStatus.begin(),deviceStatus.end());

    boost::shared_array<uint8> send_buff(new uint8[msg_buf_size]);
	CDataStream sendds(send_buff.get(), msg_buf_size);

    MsgHeader msg_header;
    msg_header.msg_id = MSG_ID_STS_SESSION_STATUS_REPORT;
    msg_header.msg_type = MSG_TYPE_REQ;
    msg_header.msg_seq = ++msg_seq_;
    sendds << msg_header;

	sendds << device_report;
    *((WORD*)sendds.getbuffer()) = sendds.size();

	int ret = tracker_agent_->Send((const unsigned char*)sendds.getbuffer(), sendds.size());
    Debug("report device status to:%s, device_num:%u, msg_size:%u, ret=%d", 
        hi_tracker_.GetNodeString().c_str(), 
        device_report.device_num, 
        sendds.size(), 
        ret );
}

void CStatusReportClient::ReportServLoad()
{
    BYTE buffer[1024];
    CDataStream sendds((unsigned char *)buffer, 1024);

    report_serv_load_ret_errcode_ = -1;
    last_report_serv_load_tick_ = get_current_tick();

    MsgHeader msg_header;
    msg_header.msg_id = MSG_ID_STS_LOAD_REPORT;
    msg_header.msg_type = MSG_TYPE_REQ;
    msg_header.msg_seq = ++msg_seq_;
    sendds << msg_header;

    sendds << load_report_;
    *((WORD*)sendds.getbuffer()) = sendds.size();
    int ret =tracker_agent_->Send((const unsigned char*)sendds.getbuffer(),sendds.size());
     Trace("report server load to:%s, msg_size:%u,ret=%d", 
        hi_tracker_.GetNodeString().c_str(), sendds.size(), ret);
}

ostringstream& CStatusReportClient::DumpInfo(ostringstream& oss)
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
    oss << last_report_dev_status_tick_;
    oss << "\"";

    oss << "}";
    return oss;
}

