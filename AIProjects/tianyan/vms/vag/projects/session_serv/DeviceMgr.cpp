#include <sys/time.h>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include "base/include/logging_posix.h"
#include "base/include/ParamParser.h"
#include "DeviceMgr.h"
#include "ServerLogical.h"

CDeviceMgr::CDeviceMgr()
{
}

CDeviceMgr::~CDeviceMgr(void)
{
}

void CDeviceMgr::Update()
{
    map<string, CDeviceContext_ptr > devices;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        devices = did_devices_;
    }

    map<string, CDeviceContext_ptr >::iterator it = devices.begin();
    for ( ; it!= devices.end(); ++it )
    {
        CDeviceContext_ptr pSessionCtx = it->second;
        pSessionCtx->Update();

        bool is_alive = pSessionCtx->IsAlive();
        if( !is_alive )
        {
            string s_did = it->first;
            CHostInfo hiRemote = pSessionCtx->GetRemote();

            Debug("form(%s) did(%s), device is not alive!", hiRemote.GetNodeString().c_str(), s_did.c_str());

            pSessionCtx->OnTcpClose(hiRemote);
            OnDeviceOffline(hiRemote, s_did);

            {
                boost::lock_guard<boost::recursive_mutex> lock(lock_);
                hi_devices_.erase(hiRemote);
                did_devices_.erase(s_did);
            }
        }
    }
}

void CDeviceMgr::DoIdleTask()
{
    return;
}

bool CDeviceMgr::OnTCPClosed( const CHostInfo& hiRemote )
{
    CDeviceContext_ptr pDevCtx = GetDeviceContext(hiRemote);
    if(!pDevCtx)
    {
        //Error("host_info(%s) find device seesion context failed!", hiRemote.GetNodeString().c_str());
        return false;
    }

    string s_did = pDevCtx->GetDeviceId();

    Debug("host_info(%s), did(%s), closed!", hiRemote.GetNodeString().c_str(), s_did.c_str());

    pDevCtx->OnTcpClose(hiRemote);
    this->OnDeviceOffline(hiRemote, s_did);

    return true;
}

CDeviceContext_ptr CDeviceMgr::GetDeviceContext(const string& s_did)
{
    CDeviceContext_ptr pDevCtx;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        map<string, CDeviceContext_ptr >::iterator it = did_devices_.find(s_did);
        if ( it != did_devices_.end() )
        {
            pDevCtx = it->second;
        }
    }
    return pDevCtx;
}

CDeviceContext_ptr CDeviceMgr::GetDeviceContext(const CHostInfo& hiRemote)
{
    CDeviceContext_ptr pSessionContext;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        map<CHostInfo,CDeviceContext_ptr >::iterator it = hi_devices_.find(hiRemote);
        if ( it != hi_devices_.end() )
        {
            pSessionContext = it->second;
        }
    }

    return pSessionContext;
}

uint32 CDeviceMgr::GetDeviceNum()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return did_devices_.size();
}

uint32 CDeviceMgr::GetConnectNum()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return hi_devices_.size();
}

void CDeviceMgr::GenDeviceStatus(vector<SDeviceSessionStatus>& deviceStatus)
{
    map<string, CDeviceContext_ptr > devices;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        devices = did_devices_;
    }

    map<string, CDeviceContext_ptr >::iterator it = devices.begin();
    for( ; it!=devices.end(); it++ )
    {
        string s_did = it->first;
        CDeviceContext_ptr pDevCtx = it->second;

        SDeviceSessionStatus ds;
        ds.mask = 0x01;
        {
            ds.did = s_did;
        }

        ds.mask |= 0x02;
        {
            ds.status = SDeviceSessionStatus::enm_dev_status_online;
            ds.timestamp = pDevCtx->GetLoginTimestamp();
        }

        ds.mask |= 0x04;
        {
            pDevCtx->GetVersion(ds.version);
            ds.dev_type = pDevCtx->GetDeviceType();
            ds.channel_num = pDevCtx->GetChannelNum();
            GetService()->GetServerHostAddr(ds.session_serv_addr.ip, ds.session_serv_addr.port);
        }
        
        ds.mask |= 0x08;
        {
            ds.channel_list_size = pDevCtx->GetChannelNum();
            pDevCtx->GetChannels(ds.channel_list);
        }

        deviceStatus.push_back(ds);
    }
}

bool CDeviceMgr::OnDeviceOffline(const CHostInfo& hiRemote, const string& s_did)
{
    CStatusReportClient_ptr pStatusAgent = GetService()->GetStatusReportClient();
    SDeviceSessionStatus ds;
    ds.mask = 0x01;
    {
        ds.did = s_did;
    }

    ds.mask |= 0x02;
    {
        ds.status = SDeviceSessionStatus::enm_dev_status_offline;

        struct timeval cur_tv;
        gettimeofday(&cur_tv, NULL);
        ds.timestamp.tv_sec = (uint64)cur_tv.tv_sec;
        ds.timestamp.tv_usec = (uint64)cur_tv.tv_usec;
    }
    pStatusAgent->PushNotifyStatus(ds);

    //clear map
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        hi_devices_.erase(hiRemote);
        did_devices_.erase(s_did);
    }

    return true;
}

bool CDeviceMgr::OnDeviceOnline(const CHostInfo& hiRemote, const string& s_did, CDeviceContext_ptr pDevCtx)
{
    CStatusReportClient_ptr pStatusAgent = GetService()->GetStatusReportClient();
    SDeviceSessionStatus ds;
    ds.mask = 0x01;
    {
        ds.did = s_did;
    }

    ds.mask |= 0x02;
    {
        ds.status = SDeviceSessionStatus::enm_dev_status_online;
        ds.timestamp = pDevCtx->GetLoginTimestamp();
    }

    ds.mask |= 0x04;
    {
        pDevCtx->GetVersion(ds.version);
        ds.dev_type = pDevCtx->GetDeviceType();
        ds.channel_num = pDevCtx->GetChannelNum();
        GetService()->GetServerHostAddr(ds.session_serv_addr.ip, ds.session_serv_addr.port);
    }

    ds.mask |= 0x08;
    {
        ds.channel_list_size = pDevCtx->GetChannelNum();
        pDevCtx->GetChannels(ds.channel_list);
    }
    (void)pStatusAgent->PushNotifyStatus(ds);
    
    //add map
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        hi_devices_[hiRemote] = pDevCtx;
        did_devices_[s_did] = pDevCtx;
    }

    return true;
}

/*
bool CDeviceMgr::MediaOpen(const SDeviceChannel& dc, const string& session_id, uint16 session_type, const SMediaDesc& desc, const vector<HostAddr>& addrs)
{
    do 
    {
        CDeviceID device_id(dc.did);
        CDeviceContext_ptr pSessionContext = GetDeviceContext(device_id);
        if (!pSessionContext)
        {
            string s_did;
            device_id.getidstring(s_did);
            Debug("cannot found session, dc:%s!",
                dc.GetString().c_str());
            break;
        }

        return pSessionContext->MediaOpen(dc, session_id, session_type, desc, addrs);

    } while (0);

    return false;
}

bool CDeviceMgr::MediaClose(const SDeviceChannel& dc, const string& session_id)
{
    do 
    {
        CDeviceID device_id(dc.did);
        CDeviceContext_ptr pSessionContext = GetDeviceContext(device_id);
        if (!pSessionContext)
        {
            Debug("cannot found session, dc:%s!",
                dc.GetString().c_str());
            break;
        }

        return pSessionContext->MediaClose(dc, session_id);

    } while (0);

    return false;
}
*/

bool CDeviceMgr::ON_DeviceLoginRequest(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceLoginReq& req, DeviceLoginResp& resp)
{
    bool ret = false;
    do
    {
        Info("login req from(%s) did(%s)", hiRemote.GetNodeString().c_str(), req.device_id.c_str());

        resp.mask = 0x01;
        resp.resp_code = EN_SUCCESS;

        if( req.mask&0x01 == 0 || req.device_id.empty() )
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            Error("from(%s), msg incorrect, mask(0x%x), did(%s)!", 
                hiRemote.GetNodeString().c_str(), 
                req.mask,
                req.device_id.c_str());
            break;
        }

        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<CHostInfo,CDeviceContext_ptr >::iterator it = hi_devices_.find(hiRemote);
            if ( it != hi_devices_.end() )
            {
                resp.resp_code = EN_DEV_ERR_ALREADY_LOGIN;
                Error("from(%s), did:%s, has already login!", hiRemote.GetNodeString().c_str(), req.device_id.c_str());
                break;
            }
        }
        
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<string, CDeviceContext_ptr >::iterator it = did_devices_.find( req.device_id );
            if ( it != did_devices_.end() )
            {
                resp.resp_code = EN_DEV_ERR_ALREADY_LOGIN;
                Error("from(%s), did:%s, has already login!", hiRemote.GetNodeString().c_str(), req.device_id.c_str());
                break;
            }
        }

        CDeviceContext_ptr pDevCtx = CDeviceContext_ptr(new CDeviceContext());
        if ( !pDevCtx )
        {
            resp.resp_code = EN_ERR_MALLOC_FAIL;
            Error("from(%s), did:%s, malloc device context failed!", hiRemote.GetNodeString().c_str(), req.device_id.c_str());
            break;
        }

        if( !pDevCtx->ON_DeviceLoginRequest(sink, hiRemote, req, resp) )
        {
            Error("from(%s), did:%s, handle login request failed!", hiRemote.GetNodeString().c_str(), req.device_id.c_str());
            break;
        }

        OnDeviceOnline(hiRemote, req.device_id, pDevCtx);

        hiRemote.GetIP( resp.public_addr.ip );
        resp.public_addr.port = hiRemote.GetPort();
        ret = true;

    } while (0);

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));

    MsgHeader header;
    header.msg_id = MSG_ID_DEV_LOGIN;
    header.msg_seq = msg_seq;
    header.msg_type = MSG_TYPE_RESP;
    sendds << header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();
    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());

    Info("send login response to (%s), did(%s), resp(0x%x, %d)", 
        hiRemote.GetNodeString().c_str(), req.device_id.c_str(), resp.resp_code, resp.resp_code);

    return ret;
}

bool CDeviceMgr::ON_DeviceAbilityReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceAbilityReportReq& req )
{
	do
	{
		CDeviceContext_ptr pDevCtx = GetDeviceContext(hiRemote);
		if (!pDevCtx)
		{
			Error("from(%s), cannot found session!", hiRemote.GetNodeString().c_str());
			break;
		}

		return	pDevCtx->ON_DeviceAbilityReport(sink, hiRemote, msg_seq, req);
	} while (false);
	return false;
}

bool CDeviceMgr::ON_StatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceStatusReportReq&req, DeviceStatusReportResp& resp)
{
    do 
    {
        CDeviceContext_ptr pDevCtx = GetDeviceContext(hiRemote);
        if (!pDevCtx)
        {
            Error("from(%s), cannot found session!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pDevCtx->ON_StatusReport(sink, hiRemote, msg_seq, req, resp);
    } while (0);
    return false;
}

bool CDeviceMgr::ON_DeviceAlarmReport(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,const DeviceAlarmReportReq& report, DeviceAlarmReportResp& resp)
{
    Debug("from(%s),recv device alarm,  did(%s), channel_id(%d), alarm_type(0x%x), alarm_status(%d).", 
        hiRemote.GetNodeString().c_str(), 
        report.device_id.c_str(),
        report.channel_id,
        report.alarm_type,
        report.alarm_status );

    resp.mask = 0;
    resp.resp_code = 0;

    char msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));

    MsgHeader header;
    header.msg_id = MSG_ID_DEV_ALARM_REPORT;
    header.msg_seq = msg_seq;
    header.msg_type = MSG_TYPE_RESP;
    sendds << header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();
    sink->SendFunc((unsigned char*)(sendds.getbuffer()),sendds.size());

    Debug("send alarm report response to (%s), did(%s), channel_id(%d), resp(0x%x, %d)", 
        hiRemote.GetNodeString().c_str(), report.device_id.c_str(), report.channel_id, resp.resp_code, resp.resp_code);

}

bool CDeviceMgr::ON_DeviceMediaOpenResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceMediaOpenResp& req)
{
    do 
    {
        CDeviceContext_ptr pDevCtx = GetDeviceContext(hiRemote);
        if (!pDevCtx)
        {
            Error("from(%s), cannot found session!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pDevCtx->ON_DeviceMediaOpenResp(sink, hiRemote, msg_seq, req);
    } while (0);
    return false;
}

bool CDeviceMgr::ON_DeviceSnapResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceSnapResp& resp)
{
    do 
    {
        CDeviceContext_ptr pDevCtx = GetDeviceContext(hiRemote);
        if (!pDevCtx)
        {
            Error("from(%s), cannot found session!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pDevCtx->ON_DeviceSnapResp(sink, hiRemote, msg_seq, resp);
    } while (0);
    return false;
}

ostringstream& CDeviceMgr::DumpDeviceInfo(const string& s_did, ostringstream& oss)
{
    CDeviceContext_ptr pSession;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        map<string, CDeviceContext_ptr >::iterator it = did_devices_.find( s_did );
        if ( it != did_devices_.end() )
        {
            pSession = it->second;
        }
    }
    if (pSession)
    {
        pSession->DumpInfo(oss);
    }	
    return oss;
}

ostringstream& CDeviceMgr::DumpInfo(ostringstream& oss, string& verbose)
{
    oss << "{" ;

    {
        map<string,CDeviceContext_ptr > devices;
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            devices = did_devices_;
        }

        oss << "\"device_num\":";
        oss << devices.size();

        if (devices.size())
        {
            oss << ",";
            oss << "\"devices\":[";
            map<string, CDeviceContext_ptr >::iterator it = devices.begin();
            if ( it != devices.end() )
            {
                it->second->DumpInfo(oss, verbose);
                ++it;
            }
            for ( ;it != devices.end(); ++it)
            {
                oss << ",";
                it->second->DumpInfo(oss, verbose);		
            }
            oss << "]";
        }
    }

    oss << "}" ;

    return oss;
}
