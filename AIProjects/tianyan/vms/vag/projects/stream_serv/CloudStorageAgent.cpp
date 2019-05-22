
#include "CloudStorageAgent.h"

CCloudStorageAgent::CCloudStorageAgent(const CHostInfo& hiServer)
	:hi_server_(hiServer)
    ,agent_status_(en_serv_agent_init)
    ,send_seq_(0)
{
}

CCloudStorageAgent::~CCloudStorageAgent(void)
{
    if(agent_)
    {
        AYClient_DestroyAYTCPClient(agent_);
        agent_ = NULL;
    }
}

void CCloudStorageAgent::SetDC(const SDeviceChannel& dc)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    dc_ = dc;
}

void CCloudStorageAgent::SetAudioInfo(protocol::AudioCodecInfo audio_info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    audio_info_ = audio_info;
}

void CCloudStorageAgent::SetVideoInfo(protocol::VideoCodecInfo video_info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    video_info_ = video_info;
}

int CCloudStorageAgent::OnStream(MI_FrameData_ptr frame_data)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    send_msg_que_.push(frame_data);
}

void CCloudStorageAgent::Update()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if(!send_msg_que_.size())
    {
        return;
    }

    switch(agent_status_)
    {
    case en_serv_agent_init:
        {
            if( StartConnect() < 0 )
            {
                agent_status_ = en_serv_agent_error;
            }
            else
            {
                agent_status_ = en_serv_agent_connecting;
            }            
            agent_status_tick_ = get_current_tick();
        }
        break;
    case en_serv_agent_connecting:
        {
            uint32 curr_tick = get_current_tick();
            if(curr_tick-agent_status_tick_ > 5*1000)
            {
                agent_status_ = en_serv_agent_error;
                agent_status_tick_ = curr_tick;
            }
        }
        break;
    case en_serv_agent_connected:
        {
            if(SendMediaConnectReq() < 0)
            {
                agent_status_ = en_serv_agent_error;
                agent_status_tick_ = get_current_tick();
            }
            else
            {
                agent_status_ = en_serv_agent_login;
                agent_status_tick_ = get_current_tick();
            }
        }
        break;
    case en_serv_agent_login:
        {
            uint32 curr_tick = get_current_tick();
            if(curr_tick-agent_status_tick_ > 2*1000)
            {
                agent_status_ = en_serv_agent_error;
                agent_status_tick_ = curr_tick;
            }
        }
        break;
    case en_serv_agent_runloop:
        {
            if( SendMediaFrameNotify() < 0 )
            {
                agent_status_ = en_serv_agent_error;
                agent_status_tick_ = get_current_tick();
            }
        }
        break;
    default:
        {
            if(agent_)
            {
                AYClient_DestroyAYTCPClient(agent_);
                agent_ = NULL;
            }
            agent_status_ = en_serv_agent_init;
            agent_status_tick_ = get_current_tick();
        }
        break;
    }
}

int CCloudStorageAgent::StartConnect()
{
    do
    {
        if(!hi_server_.IsValid())
        {
            Error("server address is invalid, server(%s)!", hi_server_.GetNodeString().c_str());
            break;
        }

        agent_ = AYClient_CreateAYTCPClient();
        if( !agent_ )
        {
            Error("create server agent failed, server(%s).", hi_server_.GetNodeString().c_str());
            break;
        }
        agent_->AdviseSink(this);

        int ret = agent_->Connect(hi_server_.GetIP(), hi_server_.GetPort()) < 0;
        if( ret < 0 )
        {
            Error("connet to server failed, server(%s), ret(%d).", hi_server_.GetNodeString().c_str(), ret);
            break;
        }
        Debug("start server agent, server(%s).", hi_server_.GetNodeString().c_str());
        return 0;
    } while (false);
    return -1;
}

int CCloudStorageAgent::SendMediaConnectReq()
{
    do 
    {
        if(agent_status_ != en_serv_agent_connected)
        {
            break;
        }

        StreamMediaConnectReq o_req;        
        {
            o_req.mask = 0x01;
            o_req.session_type = MEDIA_SESSION_TYPE_LIVE;
            o_req.session_id = dc_.GetString();
            o_req.session_media = 0x03;
            o_req.endpoint_name = "UlucuStream";
            o_req.endpoint_type = 3;
            o_req.device_id = dc_.device_id_;
            o_req.channel_id = dc_.channel_id_;
            o_req.stream_id = dc_.stream_id_;

            o_req.mask |= 0x02;
            o_req.video_codec = video_info_;
            o_req.video_direct = MEDIA_DIR_SEND_ONLY;

            o_req.mask |= 0x02;
            o_req.audio_codec = audio_info_;
            o_req.audio_direct = MEDIA_DIR_SEND_ONLY;
        }

        MsgHeader msg_header;
        msg_header.msg_id	= MSG_ID_MEDIA_CONNECT;
        msg_header.msg_type = MSG_TYPE_REQ;
        msg_header.msg_seq	= ++send_seq_;

        unsigned char msg_buffe[512];
        CDataStream sendds(msg_buffe, sizeof(msg_buffe));
        sendds << msg_header;
        sendds << o_req;
        *((WORD*)sendds.getbuffer()) = sendds.size();        
        int ret = agent_->Send((const unsigned char*)sendds.getbuffer(),sendds.size());
        if(ret < 0)
        {
            Error("send meida connect msg failed, server(%s), ret=%d", hi_server_.GetNodeString().c_str(), ret );
            break;
        }        
        Debug("send meida connect msg to server(%s).", hi_server_.GetNodeString().c_str() );
        return 0;
    }while (0);    
    return -1;
}

int CCloudStorageAgent::SendMediaFrameNotify()
{
    if(send_msg_que_.empty())
    {
        return 0;
    }

    do
    {
        int ret = -1;
        MI_FrameData_ptr frame_data = send_msg_que_.front();
        int piece_cnt = (frame_data->frame_size_ + MAX_MSG_BUFF_SIZE - 1) / MAX_MSG_BUFF_SIZE;
        for (int i = 0; i < piece_cnt; i++)
        {
            MsgHeader header;
            {
                header.msg_id = MSG_ID_MEDIA_FRAME;
                header.msg_type = MSG_TYPE_NOTIFY;
                header.msg_seq = ++send_seq_;
            }

            StreamMediaFrameNotify notify;
            {
                notify.mask = 0x01;
                notify.session_id = dc_.GetString();

                notify.mask |= 0x02;
                notify.frame_type = frame_data->frame_type_;
                notify.frame_av_seq = frame_data->frame_av_seq_;
                notify.frame_seq = frame_data->frame_seq_;
                notify.frame_ts = frame_data->frame_ts_;
                notify.frame_size = frame_data->frame_size_;

                uint32 offset = i * MAX_MSG_BUFF_SIZE;
                uint32 length = MAX_MSG_BUFF_SIZE;
                if (offset + length > notify.frame_size)
                {
                    length = notify.frame_size - offset;
                }

                notify.mask |= 0x08;
                notify.offset = offset;
                notify.data_size = length;
                for(uint32 i=0; i<length; ++i)
                {
                    uint8 data = *(frame_data->data_.get()+offset+i);
                    notify.datas.push_back(data);
                }
            }

            uint8 szSendBuff[2*MAX_MSG_BUFF_SIZE];
            CDataStream sendds(szSendBuff, sizeof(szSendBuff));
            {
                sendds << header;
                sendds << notify;
                *((WORD*)sendds.getbuffer()) = sendds.size();
            }

            ret = agent_->Send((const unsigned char*)sendds.getbuffer(), sendds.size());
            if(ret < 0)
            {
                Error(
                    "send meida frame msg failed, server(%s),ret(%d). "
                    "dc(%s), frame_type(0x%x),frame_av_seq(%u),frame_seq(%u),frame_ts(%u),frame_size(%u),offset(%u),data_size(%u)!", 
                    hi_server_.GetNodeString().c_str(),
                    ret,
                    dc_.GetString().c_str(),
                    notify.frame_type, notify.frame_av_seq, notify.frame_seq, notify.frame_ts, notify.frame_size, notify.offset, notify.data_size);
                break;
            }
        }

        if(ret<0)
        {
            break;
        }

        send_msg_que_.pop();

        return 0;
    } while (0);
    return -1;
}

int CCloudStorageAgent::OnMediaConnectResp(const StreamMediaConnectResp& resp)
{
	do
	{
        Trace("resp_code(%d), server(%s).", hi_server_.GetNodeString().c_str(), resp.resp_code);
        if(resp.resp_code != 0)
        {
            break;
        }
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        agent_status_ = en_serv_agent_runloop;
        Debug("server(%s), agent_status:(%d)", hi_server_.GetNodeString().c_str(),(int)agent_status_);
		return 0;
	} while(false);
	return -1;
}

int CCloudStorageAgent::OnTCPConnected(uint32 ip, uint16 port)
{
	Debug("server(%s), agent_status:(%d)", hi_server_.GetNodeString().c_str(),(int)agent_status_);
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
	if(agent_status_ != en_serv_agent_connecting)
	{
		agent_status_ = en_serv_agent_error;
	}
	else
	{
		agent_status_ = en_serv_agent_connected;
	}
    agent_status_tick_ = get_current_tick();
    return 0;
}

int CCloudStorageAgent::OnTCPConnectFailed(uint32 ip, uint16 port)
{
    Warn("connect failed, server(%s)!", hi_server_.GetNodeString().c_str());
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
	agent_status_ = en_serv_agent_error;
    agent_status_tick_ = get_current_tick();
    return 0;
}

int	CCloudStorageAgent::OnTCPMessage(uint32 ip, uint16 port, uint8* data, uint32 data_len)
{
    CDataStream recvds(data, data_len);
    protocol::MSG_HEADER header;
    recvds >> header;

    CHostInfo hi_remote(ip, port);
    Debug("from(%s),msg_id(0x%x), data_len(%u)", hi_remote.GetNodeString().c_str(), header.msg_id, data_len);

	int ret = -1;
	switch(header.msg_id)
	{
	case MSG_ID_MEDIA_CONNECT:
		{
            if(header.msg_type != protocol::MSG_TYPE_RESP)
            {
                ret = -101;
                break;
            }

            protocol::StreamMediaConnectResp resp;
            recvds >> resp;
            if(!recvds.good_bit())
            {
                ret = -102;
                break;
            }
			ret = OnMediaConnectResp(resp);
			break;
		}
	default:
		{
            ret = -3;
			Error("recv unknown message, server(%s)!", hi_server_.GetNodeString().c_str());
			break;
		}
	}
	return ret;
}

int CCloudStorageAgent::OnTCPClose(uint32 ip, uint16 port)
{
	Warn("tcp closed, server(%s)!", hi_server_.GetNodeString().c_str());
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    agent_status_ = en_serv_agent_error;
    agent_status_tick_ = get_current_tick();
    return 0;
}
