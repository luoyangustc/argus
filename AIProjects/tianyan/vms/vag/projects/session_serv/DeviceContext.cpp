#include <sys/time.h>
#include <fstream>
#include "logging_posix.h"
#include "DeviceContext.h"
#include "UserContext.h"
#include "ServerLogical.h"

CDeviceContext::CDeviceContext()
{
    send_sink_ = NULL;
    cmd_seq_ = 0;

    dev_type_ = 0;
    channel_num_ = 0;
    channels_.clear();
    disc_size_ = 0;
    disc_free_size_ = 0;

    sdcard_status_ = 0;

    media_trans_type_ = 1;  // 媒体传输类型：1:ES_OVER_TCP 2:ES_OVER_UDP, 默认值是1
    max_live_streams_per_ch_ = 0;
    max_playback_streams_per_ch_ = 0;
    max_playback_streams_ = 0;

    login_timestamp_.tv_sec = 0;
    login_timestamp_.tv_usec = 0;

    last_active_tick_ = 0;

    snap_tasks_.clear();
}

CDeviceContext::~CDeviceContext()
{
    Info("form(%s) did(%s), SessionContext destroy", hi_remote_.GetNodeString().c_str(), device_id_.c_str());
}

void CDeviceContext::Update()
{
    map<uint16, CSnaper_ptr> snap_tasks;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        snap_tasks = snap_tasks_;
    }

    map<uint16, CSnaper_ptr>::iterator it = snap_tasks.begin();
    for( ; it!= snap_tasks.end(); ++it )
    {
        uint16 channel_id = it->first;
        CSnaper_ptr pSnaper = it->second;
        if( pSnaper->GetStatus() < 0 || pSnaper->life_time() > 2*1000 )
        {
            string err_msg;
            if( pSnaper->GetStatus() < 0)
            {
                err_msg = "snper status err("+ boost::lexical_cast<string>(pSnaper->GetStatus()) + ")";
            }
            else
            {
                err_msg = "wait snap resp timeout!";
            }

            {
                boost::lock_guard<boost::recursive_mutex> lock(lock_);
                vector<IDeviceSnapSink_ptr>& sinks = snap_sinks_[channel_id];
                vector<IDeviceSnapSink_ptr>::iterator it = sinks.begin();
                for( ; it != sinks.end(); ++it )
                {
                    (*it)->OnDeviceSnapAck( -1, err_msg);
                }
                snap_tasks_.erase(channel_id); //clear
                snap_sinks_.erase(channel_id); //clear
            }
        }
    }
}

bool CDeviceContext::IsAlive()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if( ( get_current_tick() - last_active_tick_ ) > 3*20*1000)
    {
        return false;
    }

    return true;
}

void CDeviceContext::OnTcpClose(const CHostInfo& hiRemote)
{
    Debug("form(%s) did(%s), tcp close", 
        hi_remote_.GetNodeString().c_str(), device_id_.c_str());

    CServerLogical::GetLogical()->GetMediaSessionMgr()->OnDeviceClose( device_id_ );
}

void CDeviceContext::GetVersion(string& ver)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    ver = version_;
}

uint8 CDeviceContext::GetDeviceType()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
	return dev_type_;
}

uint8 CDeviceContext::GetChannelNum()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    return channel_num_;
}

bool CDeviceContext::GetChannel( uint16 channel_id,  protocol::DevChannelInfo& channel_info )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if( channel_num_ == 0 || channel_id > channel_num_ )
    {
        return false;
    }

    channel_info = channels_[ channel_id -1 ];

    return true;
}

void CDeviceContext::GetChannels(vector<protocol::DevChannelInfo>& channels)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    channels.clear();
    channels = channels_;
}

bool CDeviceContext::SendMessage(uint8* data_buff, uint32 data_size)
{
    if( !send_sink_ )
    {
        return false;
    }

    (void)send_sink_->SendFunc( data_buff, data_size );

    return true;
}

bool CDeviceContext::ON_DeviceLoginRequest(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, const DeviceLoginReq& req, DeviceLoginResp& resp)
{
    do
	{
        boost::lock_guard<boost::recursive_mutex> lock(lock_);

        CServerLogical* pService = GetService();
        CTokenMgr_ptr pTokenMgr = pService->GetTokenMgr();

        string s_token;
        s_token.assign((char*)req.token.token_bin, req.token.token_bin_length);

        if ( pService->IsEnableTokenAuth() )
        {
            if(!pTokenMgr->SessionDeviceToken_Auth(s_token, req.device_id) )
            {
                resp.resp_code = EN_ERR_TOKEN_CHECK_FAIL;
                Error("from(%s), did(%s), token check error!", hiRemote.GetNodeString().c_str(), req.device_id.c_str());
                break;
            }
        }

        send_sink_ = sink;
        hi_remote_ = hiRemote;

        if ( req.mask & 0x01 )
        {
            device_id_ = req.device_id;
            version_ = req.version;
            dev_type_ = req.dev_type;

            channel_num_ = req.channel_num;
            channels_ = req.channels;
            if (channel_num_ != channels_.size())
            {
                Error( "from(%s), did(%s), channel_num(%u), channels.size(%u)", 
                    hiRemote.GetNodeString().c_str(), device_id_.c_str(), channel_num_, channels_.size() );
                break;
            }

            token_ = req.token;

            Debug( "from(%s), did(%s), dev_type(%u), channel_num(%u)", 
                hiRemote.GetNodeString().c_str(), device_id_.c_str(), (uint32)dev_type_, (uint32)channel_num_ );
        }
        
        if ( req.mask & 0x02)
        {
            hi_private_addr_ = CHostInfo( req.private_addr.ip, req.private_addr.port );
            Debug("from(%s), did(%s), private_addr(%s)", 
                hiRemote.GetNodeString().c_str(), device_id_.c_str(), hi_private_addr_.GetNodeString().c_str());
        }
		
        struct timeval cur_tv;
        gettimeofday(&cur_tv, NULL);
        login_timestamp_.tv_sec = (uint64)cur_tv.tv_sec;
        login_timestamp_.tv_usec = (uint64)cur_tv.tv_usec;

        last_active_tick_ = get_current_tick();

		return true;

	} while (false);

	return false;
}

bool CDeviceContext::MediaOpen(SDeviceChannel dc, string session_id, uint16 session_type, SMediaDesc desc, vector<protocol::HostAddr> addrs)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    if(dc.channel_id_ > channel_num_)
    {
        Error("dc(%s), session_id(%s), session_type(%d), channel_id(%d) is over max_channel_id(%d).", 
            dc.GetString().c_str(), session_id.c_str(), session_type,
            (int)dc.channel_id_, (int)channel_num_);
        return false;
    }
    else if( channels_[dc.channel_id_-1].channel_status == protocol::CHANNEL_STS_OFFLINE )
    {
        Error("dc(%s), session_id(%s), session_type(%d), channel_id(%d) is offline.", 
            dc.GetString().c_str(), session_id.c_str(), session_type,
            (int)dc.channel_id_, (int)channel_num_);
        return false;
    }

    if( channels_[dc.channel_id_-1].stream_num == 0 ||
        dc.stream_id_ >= channels_[dc.channel_id_-1].stream_num )
    {
        Error("dc(%s), session_id(%s), session_type(%d), channel_id(%d),stream_num(%d), req_stream_id(%d).", 
            dc.GetString().c_str(), session_id.c_str(), session_type,
            (int)dc.channel_id_, (int)channels_[dc.channel_id_-1].stream_num, (int)dc.stream_id_);
        return false;
    }

    struct DeviceMediaOpenReq req;
    req.mask = 0;
    
    req.mask |= 0x01;
    {
        req.device_id = dc.device_id_;
        req.channel_id = dc.channel_id_;
        req.stream_id = dc.stream_id_;
        req.session_type = session_type;    // refer to 'MediaSessionType'
        req.session_id = session_id;      
        req.session_media = 0x03;   // 媒体类型：0x01:Video 0x02:Audio, 0x03:all
    }
    
    req.mask |= 0x02;
    {
        req.video_codec = channels_[dc.channel_id_-1].stream_list[dc.stream_id_].video_codec;
    }
    
    req.mask |= 0x04;
    {
        req.audio_codec = channels_[dc.channel_id_-1].audio_codec;
    }

    req.mask |= 0x08;
    {
        req.transport_type = media_trans_type_;
    }
    
    req.mask |= 0x10;
    {
        req.begin_time = desc.begin_time;
        req.end_time = desc.end_time;
    }
    
    req.mask |= 0x20;
    {
        req.stream_num = addrs.size();
        req.stream_servers.assign(addrs.begin(), addrs.end());

        string s_token;
        if ( GetService()->IsEnableTokenAuth() )
        {
            CTokenMgr_ptr pTokenMgr = GetService()->GetTokenMgr();

            bool ret = pTokenMgr->StreamDeviceToken_Gen(dc.device_id_, dc.channel_id_, 0, GetService()->GetAccessKey(), s_token);
            if (!ret)
            {
                Error("dc(%s), session_id(%s), session_type(%d), gen stream token failed.", 
                    dc.GetString().c_str(), session_id.c_str(), session_type);
                return false;
            }

            req.stream_token.token_bin_length = s_token.length();
            memcpy(req.stream_token.token_bin, s_token.c_str(), s_token.length());
        }
    }
    
    MsgHeader header;
    header.msg_id = MSG_ID_DEV_MEDIA_OPEN;
    header.msg_type = MSG_TYPE_REQ;
    header.msg_seq = ++cmd_seq_;

    uint8 msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << header;
    sendds << req;
    *((uint16*)sendds.getbuffer()) = sendds.size();

    SendMessage((uint8*)sendds.getbuffer(), sendds.size());

    Debug( "dc(%s), session_id(%s), session_type(%d) send media open command.", 
        dc.GetString().c_str(), session_id.c_str(), session_type );
    return true;
}

bool CDeviceContext::MediaClose(SDeviceChannel dc, string session_id)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    struct DeviceMediaCloseReq req;
    req.mask = 0;
    
    req.mask |= 0x01;
    {
        req.device_id = dc.device_id_;
        req.channel_id = dc.channel_id_;
        req.stream_id = dc.stream_id_;
        req.session_id = session_id;
    }
    
    MsgHeader header;
    header.msg_id = MSG_ID_DEV_MEDIA_CLOSE;
    header.msg_type = MSG_TYPE_REQ;
    header.msg_seq = ++cmd_seq_;

    uint8 msg_buf[512];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << header;
    sendds << req;
    *((uint16*)sendds.getbuffer()) = sendds.size();
    
    SendMessage((uint8*)sendds.getbuffer(), sendds.size());

    Debug("dc(%s), session_id(%s) send media close command.", dc.GetString().c_str(), session_id.c_str());

    return true;
}

bool CDeviceContext::Screenshot( const string& device_id, uint16 channel_id, IDeviceSnapSink_ptr user_sink )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    {
        if( snap_tasks_.find(channel_id) != snap_tasks_.end() )
        {
            snap_sinks_[channel_id].push_back(user_sink);
            return true;
        }

        snap_tasks_.insert(make_pair( channel_id, CSnaper_ptr( new CSnaper() )));
        snap_sinks_[channel_id].push_back(user_sink);
    }
    
    {
        struct DeviceSnapReq req;
        req.mask = 0;

        req.mask |= 0x01;
        {
            req.device_id = device_id;
            req.channel_id = channel_id;
        }
        
        MsgHeader header;
        header.msg_id = MSG_ID_DEV_SNAP;
        header.msg_type = MSG_TYPE_REQ;
        header.msg_seq = ++cmd_seq_;
        uint8 msg_buf[4096];
        CDataStream sendds(msg_buf, sizeof(msg_buf));
        sendds << header;
        sendds << req;
        *((uint16*)sendds.getbuffer()) = sendds.size();
        SendMessage((uint8*)sendds.getbuffer(), sendds.size());

        Debug( "send snap command to device, did(%s), channel_id(%u).", 
            device_id.c_str(), (uint32)channel_id );
    }

    return true;
}

bool CDeviceContext::PtzCtrl(const string& device_id, uint16 channel_id, uint16 opt_type)
{
    struct DeviceCtrlReq req;
    req.mask = 0;
    req.device_id = device_id;
    req.channel_id = channel_id;
    req.cmd_type = CMD_PTZ;
    req.mask |= 0x01;

    struct PtzCmdData ptz_cmd_data;
    memset(&ptz_cmd_data, 0, sizeof(ptz_cmd_data));
    ptz_cmd_data.opt_type = opt_type;

    char buf[512];
    CDataStream param_ds(buf, sizeof(buf));
    param_ds << ptz_cmd_data;

    req.cmd_data_size = param_ds.size();
    req.cmd_datas.reserve(req.cmd_data_size);
    for(int i=0; i<req.cmd_data_size; i++)
    {
        req.cmd_datas.push_back(param_ds.getbuffer()[i]);
    }
    req.mask |= 0x02;

    MsgHeader header;
    header.msg_id = MSG_ID_DEV_CTRL;
    header.msg_type = MSG_TYPE_REQ;
    header.msg_seq = ++cmd_seq_;

    CDataStream sendds(buf, sizeof(buf));
    sendds << header;
    sendds << req;
    *((WORD*)sendds.getbuffer()) = sendds.size();
    
    SendMessage((uint8*)sendds.getbuffer(), sendds.size());

    Debug( "send ptz command to device, did(%s), channel_id(%u), opt_type(%d).", 
        device_id.c_str(), (uint32)channel_id, opt_type );

    return true;
}

bool CDeviceContext::MgrUpdate(const string& device_id, uint16 channel_id, uint16 mgr_type)
{
    struct DeviceCtrlReq req;
    req.mask = 0;
    req.device_id = device_id;
    req.channel_id = channel_id;
    req.cmd_type = CMD_DEV_MGR_UPDATE;
    req.mask |= 0x01;

    char buf[512];
    CDataStream param_ds(buf, sizeof(buf));
    param_ds << mgr_type;

    req.cmd_data_size = param_ds.size();
    req.cmd_datas.reserve(req.cmd_data_size);
    for(int i=0; i<req.cmd_data_size; i++)
    {
        req.cmd_datas.push_back(param_ds.getbuffer()[i]);
    }
    req.mask |= 0x02;

    MsgHeader header;
    header.msg_id = MSG_ID_DEV_CTRL;
    header.msg_type = MSG_TYPE_REQ;
    header.msg_seq = ++cmd_seq_;

    CDataStream sendds(buf, sizeof(buf));
    sendds << header;
    sendds << req;
    *((WORD*)sendds.getbuffer()) = sendds.size();

    SendMessage((uint8*)sendds.getbuffer(), sendds.size());

    Debug( "send device mgr update command to device, did(%s), channel_id(%u), mgr_type(%d).", 
        device_id.c_str(), (uint32)channel_id, mgr_type );

    return true;
}

bool CDeviceContext::ON_StatusReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceStatusReportReq& report, DeviceStatusReportResp& resp)
{
    Trace("from(%s), did(%s), mask(%d),device status report.", hiRemote.GetNodeString().c_str(), device_id_.c_str(), report.mask);

    vector<DeviceMediaSessionStatus> media_session_staus_list;

    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        
        last_active_tick_ = get_current_tick();

        if (report.mask & 0x01)
        {
            vector<protocol::DevChannelInfo>::const_iterator it = report.channels.begin();
            for ( ; it != report.channels.end(); ++it )
            {
                if( it->channel_id > channel_num_ )
                {
                    Error("from(%s), did(%s), channel_id incorrect, (%u>%u).", 
                        hiRemote.GetNodeString().c_str(), device_id_.c_str(), (uint32)it->channel_id, (uint32)channel_num_);
                    continue;
                }
                channels_[ it->channel_id - 1 ] = *it;  //Update channel status
            }

            (void)OnChanenlStatusUpate();
        }

        if (report.mask & 0x02)
        {
            media_session_staus_list = report.media_sessions;
        }

        if(report.mask & 0x04)
        {	
            sdcard_status_ = report.sdcard_status;
        }
    }

    //response device
    resp.mask = 0;
    resp.resp_code = EN_SUCCESS;
    resp.expected_cycle = 30;
    resp.mask |= 0x01;

    MsgHeader header;
    header.msg_id = MSG_ID_DEV_STATUS_REPORT;
    header.msg_type = MSG_TYPE_RESP;
    header.msg_seq = msg_seq;

    uint8 msg_buf[256];
    CDataStream sendds(msg_buf, sizeof(msg_buf));
    sendds << header;
    sendds << resp;
    *((uint16*)sendds.getbuffer()) = sendds.size();

    SendMessage((uint8*)sendds.getbuffer(), sendds.size());

    //notify media session status to MediaSessionMgr
    GetService()->GetMediaSessionMgr()->OnDeviceStatusReport(media_session_staus_list);

    return true;
}

bool CDeviceContext::ON_DeviceAbilityReport(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceAbilityReportReq& req)
{
    char debug_buff[512]={0};
    int debug_size = 0;

	do 
	{
        boost::lock_guard<boost::recursive_mutex> lock(lock_);

        debug_size = snprintf( debug_buff, sizeof(debug_buff) -1, 
                            "from(%s), did(%s), mask(%u) ", 
                            hiRemote.GetNodeString().c_str(), device_id_.c_str(), req.mask );
        if (req.mask & 0x01)
        {
            media_trans_type_ = req.media_trans_type;
            max_live_streams_per_ch_ = req.max_live_streams_per_ch;
            max_playback_streams_per_ch_ = req.max_playback_streams_per_ch;
            max_playback_streams_ = req.max_playback_streams;

            debug_size += snprintf( debug_buff+debug_size, sizeof(debug_buff) - 1 - debug_size,
                                ", media_transfer_type(%d), max_live_streams_per_ch(%d), max_playback_streams_per_ch(%d), max_playback_streams(%d) ", 
                                req.media_trans_type,
                                req.max_live_streams_per_ch,
                                req.max_playback_streams_per_ch,
                                req.max_playback_streams );
            
        }

        if (req.mask & 0x02)
        {
            disc_size_ = req.disc_size;
            disc_free_size_ = req.disc_free_size;
            debug_size += snprintf( debug_buff+debug_size, sizeof(debug_buff) - 1 - debug_size,
                                ",disc_size(%d), disc_free_size(%d)", 
                                req.disc_size, req.disc_free_size );
        }
        debug_buff[debug_size] = '0';

        //response device
        DeviceAbilityReportResp resp;
        resp.mask = 0;
        resp.resp_code = EN_SUCCESS;

        MsgHeader header;
        header.msg_id = MSG_ID_DEV_ABILITY_REPORT;
        header.msg_type = MSG_TYPE_RESP;
        header.msg_seq = msg_seq;

        uint8 msg_buf[256];
        CDataStream sendds(msg_buf, sizeof(msg_buf));
        sendds << header;
        sendds << resp;
        *((uint16*)sendds.getbuffer()) = sendds.size();

        SendMessage((uint8*)sendds.getbuffer(), sendds.size());

        Debug( "%s", debug_buff );

        return true;

	} while (false);
	return false;
}

bool CDeviceContext::ON_DeviceMediaOpenResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceMediaOpenResp& req)
{
    do
    {
        Error( "from(%s), recv device meida open response, did(%s), mask(%d), resp_code(%d), session_id(%s), stream_host(%s:%d).", 
            hiRemote.GetNodeString().c_str(), 
            device_id_.c_str(), 
            req.mask, 
            req.resp_code,
            req.session_id.c_str(), 
            req.stream_server.ip.c_str(), 
            req.stream_server.port );

        if (req.mask & 0x01)
        {
            CServerLogical::GetLogical()->GetMediaSessionMgr()->OnDeviceMediaOpenAck(req);
        }
    }
    while(false);

    return true;
}

bool CDeviceContext::ON_DeviceSnapResp(ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const DeviceSnapResp& resp)
{
    do 
    {
        CSnaper_ptr pSnaper;
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map<uint16, CSnaper_ptr>::iterator it = snap_tasks_.find(resp.channel_id);
            if( it == snap_tasks_.end() )
            {
                Warn( "from(%s), cannot find snap task, did(%s), channel_id(%d)", 
                    hiRemote.GetNodeString().c_str(), resp.device_id.c_str(), (int)resp.channel_id );
                break;
            }
            pSnaper = it->second;
        }

        {
            if( resp.resp_code != EN_SUCCESS )
            {
                {
                    boost::lock_guard<boost::recursive_mutex> lock(lock_);
                    vector<IDeviceSnapSink_ptr>& sinks = snap_sinks_[resp.channel_id];
                    vector<IDeviceSnapSink_ptr>::iterator it = sinks.begin();
                    for( ; it != sinks.end(); ++it )
                    {
                        (*it)->OnDeviceSnapAck( resp.resp_code, "device response error!");
                    }
                    snap_tasks_.erase(resp.channel_id); //clear
                    snap_sinks_.erase(resp.channel_id); //clear
                }
                Error( "from(%s), device snap error, did(%s), channel_id(%d), resp_code(%d)!", 
                    hiRemote.GetNodeString().c_str(), 
                    resp.device_id.c_str(), 
                    (int)resp.channel_id, 
                    resp.resp_code );
                break;
            }

            
            if( !pSnaper->OnDeviceSnaperResp(resp) )
            {
                {
                    boost::lock_guard<boost::recursive_mutex> lock(lock_);
                    vector<IDeviceSnapSink_ptr>& sinks = snap_sinks_[resp.channel_id];
                    vector<IDeviceSnapSink_ptr>::iterator it = sinks.begin();
                    for( ; it != sinks.end(); ++it )
                    {
                        (*it)->OnDeviceSnapAck( -1, "session server handle error!");
                    }
                    snap_tasks_.erase(resp.channel_id); //clear
                    snap_sinks_.erase(resp.channel_id); //clear
                }

                Error( "from(%s), snap failed, did(%s), channel_id(%d), err_code(%d)", 
                    hiRemote.GetNodeString().c_str(), 
                    resp.device_id.c_str(), 
                    (int)resp.channel_id,
                    pSnaper->GetStatus() );
                break;
            }

            if( !pSnaper->IsFull() )
            {
                break;
            }
        }
        
        //snap data recv complete
        string pic_save_path = GetService()->GetServCfg()->GetSnapPicSavePath();
        string pic_full_name = pic_save_path + pSnaper->pic_name_;
        ofstream out(pic_full_name.c_str());
        out.write((char*)pSnaper->datas_.get(), pSnaper->pic_size_);
        out.close();

        string pic_url =  GetService()->GetServCfg()->GetSnapPicUrl() + pSnaper->pic_name_;

        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            vector<IDeviceSnapSink_ptr>& sinks = snap_sinks_[resp.channel_id];
            vector<IDeviceSnapSink_ptr>::iterator it = sinks.begin();
            for( ; it != sinks.end(); ++it )
            {
                (*it)->OnDeviceSnapAck( 0, "success", pic_url);
            }
            snap_tasks_.erase(resp.channel_id); //clear
            snap_sinks_.erase(resp.channel_id); //clear
        }

        Debug( "from(%s), device snap pic complete, did(%s), channel_id(%d)", 
            hiRemote.GetNodeString().c_str(), 
            pSnaper->device_id_.c_str(), 
            (int)pSnaper->channel_id_ );

    } while (0);

    return true;
}

bool CDeviceContext::OnChanenlStatusUpate()
{
    CStatusReportClient_ptr pStatusAgent = GetService()->GetStatusReportClient();
    SDeviceSessionStatus ds;
    ds.mask = 0x01;
    {
        ds.did = device_id_;
    }

    ds.mask |= 0x08;
    {
        ds.channel_list_size = channel_num_;
        this->GetChannels(ds.channel_list);
    }
    (void)pStatusAgent->PushNotifyStatus(ds);

    return true;
}

ostringstream& CDeviceContext::DumpInfo(ostringstream& oss, string verbose)
{
	return oss;
}

