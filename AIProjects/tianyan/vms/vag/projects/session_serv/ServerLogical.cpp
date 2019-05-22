#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/scoped_array.hpp>
#include <curl/curl.h>
#include <pthread.h>
#include "web_request.h"
#include "DeviceID.h"
#include "ConfigHelper.h"
#include "logging_posix.h"
#include "ParamParser.h"
#include "tick.h"
#include "AYServerApi.h"
#include "ServerLogical.h"

CServerLogical CServerLogical::logic_;

CServerLogical* GetService()
{
    return CServerLogical::GetLogical();
}

IServerLogical * IServerLogical::GetInstance()
{
	return (CServerLogical::GetLogical());
}

CServerLogical::CServerLogical()
{
    last_update_load_tick_ = 0;
}

CServerLogical::~CServerLogical()
{

}

bool CServerLogical::InitCfg(const string& strFileName)
{
	pServCfg_ = CServerCfg_ptr(new CServerCfg());
	if (!pServCfg_)
	{
		Error("new CStreamCfg_ptr failed!\n");
		return false;
	}

	if (0 != pServCfg_->ReadCfgFile(strFileName))
	{
		Error("read cfg file(%) failed!\n", strFileName.c_str());
		return false;
	}

	return true;
}

bool CServerLogical::Start()
{
    WebRequest::instance().start();

    {
        login_req_.mask = 0x01;
        login_req_.ep_type = EP_SMS;
        login_req_.http_port = pServCfg_->GetHttpPort();
        login_req_.serv_port = pServCfg_->GetServPort();

		login_req_.mask |= 0x02;
		login_req_.listen_ips = pServCfg_->GetListenIpList();
        login_req_.listen_ip_num = login_req_.listen_ips.size();
    }

    global_configitem_.enable_relay = ::GetPrivateProfileInt("setting", "enable_relay", 0, CConfigHelper::get_default_config_filename().c_str()) ? true : false;

	pStatusReportClient_ = CStatusReportClient_ptr( new CStatusReportClient( pServCfg_->GetStatusServ(),login_req_,load_report_) );

    pDeviceMgr_ = CDeviceMgr_ptr(new CDeviceMgr());
	pUserMgr_ = CUserMgr_ptr(new CUserMgr());
	pTokenMgr_ = CTokenMgr_ptr (new CTokenMgr());

    if ( pServCfg_->IsTokenCheck() )
    {
        pTokenMgr_->SetKey( pServCfg_->GetAccessKey(), pServCfg_->GetSecretKey() );
    }

    pMediaSessionMgr_ = CMediaSessionMgr_ptr(new CMediaSessionMgr());

	pCommonThreadGroup_ = CCommon_Thread_Group_ptr(new CCommon_Thread_Group());
	if(pCommonThreadGroup_ != NULL)
		pCommonThreadGroup_->start();

    pSysMonitorThread_ = CSysMonitorThread_ptr(new CSysMonitorThread());
    if (pSysMonitorThread_){
        pSysMonitorThread_->Start();
    }

    start_tick_ = get_current_tick();

	return true;
}

void CServerLogical::Stop()
{
	pStatusReportClient_.reset();
	pDeviceMgr_.reset();
	pUserMgr_.reset(); 
    WebRequest::instance().stop();
}

void CServerLogical::GetServerHostAddr(string& ip, uint16& port)
{
    ip = pServCfg_->GetServIp();
    port = pServCfg_->GetServPort();
}

void CServerLogical::UpdateLoadReport()
{
    load_report_.mask = 0x00;

    int32 tcp_conn = 0;
    int32 cpu_percent = 0;
    int32 mem_percent = 0;

    load_report_.tcp_conn_num = (WORD)tcp_conn;
    load_report_.cpu_use = (BYTE)cpu_percent;
    load_report_.memory_use = (BYTE)mem_percent;

    load_report_.mask |= 0x01;

    last_update_load_tick_ = GetTickCount();
}

void CServerLogical::Update()
{
    pSysMonitorThread_->UpdateActiveTick();

    if( GetTickCount() - last_update_load_tick_ > 10000 )
    {
        UpdateLoadReport();
    }

    {
        pUserMgr_->Update();
    }

    {
        pDeviceMgr_->Update();
    }
    
    {
        pStatusReportClient_->Update();
    }
    
    {
        pMediaSessionMgr_->Update();
    }
}

void CServerLogical::DoIdleTask()
{
    static int last_read_configfile_tick = 0;
    if (get_current_tick() - last_read_configfile_tick >= 10*1000)
    {
        string loglevel = "DEBUG"; //若配置文件中没配置日志级别，默认为DEBUG日志级别
        char loglevel_buf[128] = {0};
        ::GetPrivateProfileString("setting", "log_level", loglevel.c_str(), loglevel_buf, sizeof(loglevel_buf), CConfigHelper::get_default_config_filename().c_str());
        loglevel = loglevel_buf;

        if (strcasecmp(getloglevelstr(), loglevel.c_str()) != 0)
        {
            setloglevel(loglevel.c_str());
        }

        last_read_configfile_tick = get_current_tick(); 
    }
}

int32 CServerLogical::OnTCPMessage(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_id, CDataStream& recvds,CDataStream& sendds)
{
    do
    {
        if( !pUserMgr_ || !pTokenMgr_)
        {
            Error("from(%s), msgid(0x%x), %p,%p!",
                hiRemote.GetNodeString().c_str(), msg_id,
                pUserMgr_.get(), pTokenMgr_.get());
            break;
        }

        if( !sink )
        {
            Error("from(%s), msgid(0x%x), sink is nil!", 
                hiRemote.GetNodeString().c_str(), msg_id);
            break;
        }

        MsgHeader msg_header;
        recvds >> msg_header;

        switch(msg_header.msg_id)
        {
            //messages between device and session server
        case MSG_ID_DEV_LOGIN:
            {
                if( ON_DeviceLoginRequest(sink,hiRemote,msg_header.msg_seq,recvds,sendds) )
                {
                    return 0;
                }
            }
            break;
        case MSG_ID_DEV_ABILITY_REPORT:
            {
                if ( ON_DeviceAbilityReport(sink,hiRemote,msg_header.msg_seq,recvds,sendds) )
                {
                    return 0;
                }
            }
            break;
        case MSG_ID_DEV_STATUS_REPORT:
            {
                if( ON_StatusReport(sink,hiRemote,msg_header.msg_seq,recvds,sendds) )
                {
                    return 0;
                }
            }
            break;
        case MSG_ID_DEV_MEDIA_OPEN:
            {
                if (ON_DeviceMediaOpenResp(sink,hiRemote,msg_header.msg_seq,recvds,sendds))
                {
                    Debug("from(%s), msgid(0x%x), msg_seq(%u)", hiRemote.GetNodeString().c_str(), msg_id, msg_header.msg_seq);
                    return 0;
                }
            }
            break;
        case MSG_ID_DEV_MEDIA_CLOSE:
            {
                Debug("from(%s), msgid(0x%x), msg_seq(%u)", hiRemote.GetNodeString().c_str(), msg_id, msg_header.msg_seq);
                return 0;
            }
            break;
        case MSG_ID_DEV_SNAP:
            {
                if ( (msg_header.msg_type == MSG_TYPE_RESP) && ON_DeviceSnapResp(sink,hiRemote,msg_header.msg_seq,recvds,sendds) )
                {
                    Debug("from(%s), msgid(0x%x), msg_seq(%u)", hiRemote.GetNodeString().c_str(), msg_id, msg_header.msg_seq);
                    return 0;
                }
            }
            break;
        case MSG_ID_DEV_CTRL:
            {
                Debug("from(%s), msgid(0x%x), msg_seq(%u)", hiRemote.GetNodeString().c_str(), msg_id, msg_header.msg_seq);
                return 0;
            }
            break;
            //messages between client and session server
        case MSG_ID_CU_LOGIN:
            {
                if( ON_CuLoginRequest(sink,hiRemote,msg_header.msg_seq,recvds,sendds) )
                {
                    return 0;
                }
            }
            break;
        case MSG_ID_CU_STATUS_REPORT:
            {
                if( ON_CuStatusReport(sink,hiRemote,msg_header.msg_seq,recvds,sendds) )
                {
                    return 0;
                }
            }
            break;
        case MSG_ID_CU_MEDIA_OPEN:
            {
                if( ON_CuMediaOpen(sink, hiRemote, msg_header.msg_seq, recvds, sendds) )
                {
                    return 0;
                }
            }
            break;
        case MSG_ID_CU_MEDIA_CLOSE:
            {
                if( ON_CuMediaClose(sink,hiRemote,msg_header.msg_seq,recvds,sendds) )
                {
                    return 0;
                }
            }
            break;
        default:
            {
                Warn("invalid message(0x%x,%s), msg_size(%d)!", msg_header.msg_id, hiRemote.GetNodeString().c_str(), msg_header.msg_size);
            }
            break;
        };	
    }
    while(false);

    Warn("handle message(0x%x,%s) fail!",msg_id, hiRemote.GetNodeString().c_str());
    return -1;
}

int32 CServerLogical::OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& hiRemote,CDataStream& sendds)
{
	do
	{
		Debug("message(%x,%u)!",
			hiRemote.IP,hiRemote.Port);	
		return 0;
	}
	while(false);
	return -1;
}

int32 CServerLogical::OnTCPClosed(ITCPSessionSendSink*sink, CHostInfo& hiRemote)
{
    do 
    {
        if (pUserMgr_ && pUserMgr_->OnTCPClosed(hiRemote) )
        {
            break;
        }

        if(pDeviceMgr_ && pDeviceMgr_->OnTCPClosed(hiRemote) )
        {
            break;
        }

        /*if(pMediaSessionMgr_)
        {
            pMediaSessionMgr_->OnTCPClosed(hiRemote);
        }*/

    } while (0);
    
    return 0;
}

CDeviceContext_ptr CServerLogical::GetDeviceContext(const CHostInfo& hi )
{
    return pDeviceMgr_->GetDeviceContext(hi);
}

CDeviceContext_ptr  CServerLogical::GetDeviceContext(const string& device_id )
{
    return pDeviceMgr_->GetDeviceContext(device_id);
}

CUserContext_ptr CServerLogical::GetUserContext(const CHostInfo& hi )
{
    return pUserMgr_->GetUserContext(hi);
}

void CServerLogical::GetUserContext(const string& user_name, OUT vector<CUserContext_ptr>& user_contexts )
{
    return pUserMgr_->GetUserContextsByUsername(user_name, user_contexts);
}

MediaSession_ptr CServerLogical::GetMediaSession(const string& session_id)
{
    
}

MediaSession_ptr CServerLogical::GetMediaSession(const SDeviceChannel& dc, SessionType session_type)
{
    
}