#include "ServerLogical.h"
#include "curl/curl.h"
static const int DIFF_LOAD_EXPORTED_TICK_LAST	= 10*1000;

CServerLogical* GetService()
{
    return CServerLogical::GetLogical();
}

CServerLogical CServerLogical::logic_;

IServerLogical * IServerLogical::GetInstance()
{
    return (CServerLogical::GetLogical());
}

#ifdef USE_OPENSSL

static pthread_mutex_t *lockarray;
#include <openssl/crypto.h>

static void lock_callback(int mode, int type, char *file, int line)
{
    (void)file;
    (void)line;
    if (mode & CRYPTO_LOCK) {
        pthread_mutex_lock(&(lockarray[type]));
    } 
    else {
        pthread_mutex_unlock(&(lockarray[type]));
    }
}

static unsigned long thread_id(void)
{
    unsigned long ret;

    ret=(unsigned long)pthread_self();
    return(ret);
}

static void init_locks(void)
{
    int i;

    lockarray=(pthread_mutex_t *)OPENSSL_malloc(CRYPTO_num_locks() *
        sizeof(pthread_mutex_t));
    for (i=0; i<CRYPTO_num_locks(); i++) {
        pthread_mutex_init(&(lockarray[i]),NULL);
    }

    CRYPTO_set_id_callback((unsigned long (*)())thread_id);
    CRYPTO_set_locking_callback((void (*)())lock_callback);
}

static void kill_locks(void)
{
    int i;

    CRYPTO_set_locking_callback(NULL);
    for (i=0; i<CRYPTO_num_locks(); i++)
        pthread_mutex_destroy(&(lockarray[i]));

    OPENSSL_free(lockarray);
}

#endif

CServerLogical::CServerLogical()
    : tick_start_( 0 )
{
    CURLcode curl_code;
    curl_code = curl_global_init(CURL_GLOBAL_ALL);
    if (curl_code != CURLE_OK)
    {
        exit(0);
    }
#ifdef USE_OPENSSL
    init_locks();
#endif
}

CServerLogical::~CServerLogical()
{
#ifdef USE_OPENSSL
    kill_locks();
#endif

    curl_global_cleanup();
}

bool CServerLogical::InitCfg(const string& strFileName)
{
    pSevrCfg_ = CStreamCfg_ptr(new CStreamCfg());
    if (!pSevrCfg_)
    {
        Error("new CStreamCfg_ptr failed!\n");
        return false;
    }

    if (0 != pSevrCfg_->ReadCfgFile(strFileName))
    {
        Warn("read cfg file(%) failed!\n", strFileName.c_str());
    }

    return true;
}

bool CServerLogical::Start()
{
    do 
    {
        //pDBWriter_ = CDBWriter_ptr(new CDBWriter(pSevrCfg_->GetRecordPath()));
        pTokenMgr_ = CTokenMgr_ptr (new CTokenMgr());
        if(!pTokenMgr_)
        {
            Error("new CTokenMgr obj failed!");
            break;
        }
        pTokenMgr_->SetKey(pSevrCfg_->GetAccessKey(), pSevrCfg_->GetSecretKey());

        pMediaSessionMgr_ = CMediaSessionMgr_ptr(new CMediaSessionMgr());
        if(!pMediaSessionMgr_)
        {
            Error("new CMediaSessionMgr obj failed!");
            break;
        }

        pSysMonitorThread_ = CSysMonitorThread_ptr(new CSysMonitorThread());
        if (!pSysMonitorThread_)
        {
            Error("new CSysMonitorThread obj failed!");
            break;
        }
        if( pSysMonitorThread_->Start() == false )
        {
            Error("System monitor thread start failed!");
            break;
        }

        //pTSEncodeEngine_ = CTSEncodeEngineFactory::CreateTSEncodeEngine(this);
        InitLoginStatus();
        pStreamStatusReport_ = CStreamStatusReport_ptr(new CStreamStatusReport(pSevrCfg_->GetStatusServ(),login_req_, load_report_));
        if(!pStreamStatusReport_)
        {
            Error("new CStreamStatusReport obj failed!");
            break;
        }
        tick_start_ = get_current_tick();
        Debug("Start success!");
        return true;
    } while (0);
    return false;
}

void CServerLogical::Stop()
{
    pSevrCfg_.reset();
    //pDBWriter_.reset();
    pTokenMgr_.reset();
    pMediaSessionMgr_.reset();
    pSysMonitorThread_.reset();
	pStreamStatusReport_.reset();
}

void CServerLogical::Update()
{
    if (pSysMonitorThread_)
    {
        pSysMonitorThread_->UpdateActiveTick();
    }

	if (pMediaSessionMgr_)
    {
        pMediaSessionMgr_->Update();
    }

	static tick_t last_update_load_tick = 0;
	if (get_current_tick()-last_update_load_tick>DIFF_LOAD_EXPORTED_TICK_LAST)
	{
		UpdateLoadReport();
		last_update_load_tick = get_current_tick();
	}

    if (pStreamStatusReport_)
	{
		pStreamStatusReport_->Update();
    }
}

void CServerLogical::InitLoginStatus()
{
    login_req_.mask = 0x01;
    login_req_.ep_type = EP_STREAM;
    login_req_.http_port = pSevrCfg_->GetHttpPort();
    login_req_.serv_port = pSevrCfg_->GetServPort();

    std::vector<string> ip_list = pSevrCfg_->GetListenIpList();
    for(std::vector<string>::iterator it=ip_list.begin(); it!=ip_list.end(); it++)
    {
        login_req_.listen_ips.push_back(*it);
    }
    login_req_.listen_ip_num = login_req_.listen_ips.size();
    login_req_.mask |= 0x02;
}

void CServerLogical::UpdateLoadReport()
{
	int32 tcp_conn = 0;
	int32 cpu_percent = 0;
	int32 mem_percent = 0;

	load_report_.mask = 0x00;
	load_report_.tcp_conn_num = (WORD)tcp_conn;
	load_report_.cpu_use = (BYTE)cpu_percent;
	load_report_.memory_use = (BYTE)mem_percent;

	load_report_.mask |= 0x01;
}

void CServerLogical::DoIdleTask()
{
    /*if(pDBWriter_)
    {
        pDBWriter_->update();
    }*/

    if(pSevrCfg_)
    {
        pSevrCfg_->Update();
    }
}

int32 CServerLogical::OnUDPMessage(CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds, IN int thread_index, uint8 algo)
{
    return 0;
}

int32 CServerLogical::OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& hiRemote,CDataStream& sendds)
{
    do 
    {
        Trace("from(%s)!", hiRemote.GetNodeString().c_str());

    } while (0);

    return 0;
}

int32 CServerLogical::OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& hiRemote)
{
    Trace("from(%s)!", hiRemote.GetNodeString().c_str());

    do 
    {
        if ( !pMediaSessionMgr_ )
        {
            break;
        }
        pMediaSessionMgr_->OnTCPClosed(hiRemote);

    } while (0);

    return 0;
}

int32 CServerLogical::OnTCPMessage(ITCPSessionSendSink*sink, CHostInfo& hiRemote, uint32 msg_id, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        if (!sink)
        {
            Error("from(%s), send sink is nil!\n", hiRemote.GetNodeString().c_str() );
            break;
        }

        protocol::MsgHeader header;
        recvds >> header;
        if( !recvds.good_bit() )
        {
            Error("from(%s), parse msg error!\n", hiRemote.GetNodeString().c_str() );
            break;
        }

        Trace( "from(%s), recv msg, size(%u),type(%u), id(0x%x), seq(%u)!\n", 
            hiRemote.GetNodeString().c_str(),
            header.msg_size,
            header.msg_type,
            header.msg_id,
            header.msg_seq );

        switch(header.msg_type)
        {
        case protocol::MSG_TYPE_REQ:
            {
                return HandleReqMsg( sink, hiRemote, header, recvds, sendds);
            }
            break;
        case protocol::MSG_TYPE_RESP:
            {
                return HandleRespMsg( sink, hiRemote, header, recvds, sendds);
            }
            break;
        case protocol::MSG_TYPE_NOTIFY:
            {
                return HandleNotifyMsg( sink, hiRemote, header, recvds, sendds);
            }
            break;
        default:
            {
                Warn("invalid message(%x, %s)!", header.msg_type, hiRemote.GetNodeString().c_str());
            }
            break;
        }

    } while (false);

    Warn("from(%s), handle msg failed, msg_id(0x%x)!", hiRemote.GetNodeString().c_str(), msg_id);

    return -1;
}

int32 CServerLogical::HandleReqMsg(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds,CDataStream& sendds)
{
    switch(header.msg_id)
    {
    case protocol::MSG_ID_MEDIA_CONNECT:
        {
            if (OnConnect(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_DISCONNECT:
        {
            if (OnDisconnect(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_PLAY:
        {
            if (OnPlayReq(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_PAUSE:
        {
            if (OnPauseReq(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_STATUS:
        {
            if (OnStatusReport(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_CMD:
        {
            if (OnCmdReq(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    default:
        Warn("from(%s), invalid msg, msg_id(0x%x)!", hiRemote.GetNodeString().c_str(), header.msg_id);
        break;
    }
    Warn("from(%s), handle msg failed, msg_id(0x%x)!", hiRemote.GetNodeString().c_str(), header.msg_id );
    return -1;
}

int32 CServerLogical::HandleRespMsg(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds,CDataStream& sendds)
{
    switch(header.msg_id)
    {
    case protocol::MSG_ID_MEDIA_PLAY:
        {
            if (OnPlayResp(sink, hiRemote, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_PAUSE:
        {
            if (OnPauseResp(sink, hiRemote, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_CMD:
        {
            if (OnMediaCmdResp(sink, hiRemote, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case protocol::MSG_ID_MEDIA_CLOSE:
        {
            if (OnCloseResp(sink, hiRemote, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    default:
        Warn("from(%s), invalid msg, msg_id(0x%x)!", hiRemote.GetNodeString().c_str(), header.msg_id);
        break;
    }
    Warn("from(%s), handle msg failed, msg_id(0x%x)!", hiRemote.GetNodeString().c_str(), header.msg_id );
    return -1;
}

int32 CServerLogical::HandleNotifyMsg(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds,CDataStream& sendds)
{
    switch(header.msg_id)
    {
    case protocol::MSG_ID_MEDIA_FRAME:
        {
            if (OnFrameNotify(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    case MSG_ID_MEDIA_EOS:
        {
            if (OnEosNotify(sink, hiRemote, header, recvds, sendds))
            {
                return 0;
            }
        }
        break;
    default:
        Warn("from(%s), invalid msg, msg_id(0x%x)!", hiRemote.GetNodeString().c_str(), header.msg_id);
        break;
    }
    Warn("from(%s), handle msg failed, msg_id(0x%x)!", hiRemote.GetNodeString().c_str(), header.msg_id );
    return -1;
}

bool CServerLogical::OnConnect(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaConnectReq req;
        recvds >> req;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnConnect(sink, hiRemote, header, req);
    } while(false);
    return false;
}

bool CServerLogical::OnDisconnect(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaDisconnectReq req;
        recvds >> req;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnDisconnect(sink, hiRemote, header, req);
    } while(false);
    return false;
}

bool CServerLogical::OnStatusReport(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaStatusReq req;
        recvds >> req;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnStatusReport(sink, hiRemote, header, req);
    } while(false);
    return false;
}

bool CServerLogical::OnPlayReq(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaPlayReq req;
        recvds >> req;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnPlayReq(sink, hiRemote, header, req);
    } while(false);
    return false;
}

bool CServerLogical::OnPauseReq(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaPauseReq req;
        recvds >> req;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnPauseReq(sink, hiRemote, header, req);
    } while(false);
    return false;
}

bool CServerLogical::OnCmdReq(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaCmdReq req;
        recvds >> req;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnCmdReq(sink, hiRemote, header, req);
    } while(false);
    return false;
}

bool CServerLogical::OnPlayResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaPlayResp resp;
        recvds >> resp;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnPlayResp(sink, hiRemote, resp);
    } while(false);
    return false;
}

bool CServerLogical::OnPauseResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaPauseResp resp;
        recvds >> resp;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnPauseResp(sink, hiRemote, resp);
    } while(false);
    return false;
}

bool CServerLogical::OnMediaCmdResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaCmdResp resp;
        recvds >> resp;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnMediaCmdResp(sink, hiRemote, resp);
    } while(false);
    return false;
}

bool CServerLogical::OnCloseResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaCloseResp resp;
        recvds >> resp;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnCloseResp(sink, hiRemote, resp);
    } while(false);
    return false;
}

bool CServerLogical::OnFrameNotify(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaFrameNotify notify;
        recvds >> notify;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnFrameNotify(sink, hiRemote, header, notify);
    } while(false);
    return false;
}

bool CServerLogical::OnEosNotify(ITCPSessionSendSink*sink, CHostInfo& hiRemote, const MsgHeader& header, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        protocol::StreamMediaEosNotify notify;
        recvds >> notify;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }

        return pMediaSessionMgr_->OnEosNotify(sink, hiRemote, header, notify);
    } while(false);
    return false;
}
