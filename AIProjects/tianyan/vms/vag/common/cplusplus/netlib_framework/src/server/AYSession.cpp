#include "AYSession.h"
#include "HostInfo.h"
#include "datastream.h"
#include "tick.h"
#include "protocol_exchangekey.h"
#include "Log.h"


CAYSession::CAYSession(boost::asio::io_service& io_service)
    :CTCPHandler(io_service)
    ,m_bInTasking(false)
{
}

CAYSession::~CAYSession()
{
    DEBUG_LOG( "Session(%p), destory.", this );
}

boost::shared_ptr<CAYSession> CAYSession::SharedFromThis()
{
    return boost::dynamic_pointer_cast<CAYSession>(shared_from_this());
}

int CAYSession::Init(CTCPServer<CAYSession>* pServ)
{
    int ret = 0;
    do
    {
        if( !pServ )
        {
            ERROR_LOG( "Session(%p), server is nil!", this );
            break;
        }
        m_pServer = pServ;
        //Set tcp aysnc mode
        SetMode(en_tcp_on_async);

        //Open tcp handle
        if( (ret = Open()) < 0 )
        {
            ERROR_LOG( "Session(%p), open failed, ret=%d!", this, ret );
            break;
        }

        DEBUG_LOG( "Session(%p), success.", this );

        return 0;

    } while (0);

    return -1;
}

int CAYSession::TaskMain()
{
    int ret = 0;
    uint32 task_time = 0;
    uint32 start_tick = get_current_tick();
    DEBUG_LOG("Session(%p)-->start...", this);
    do
    {
        if( !IsOpen() )
        {
            CloseTask();
            ret = -1;
            break;
        }

#if 0
        if( RecvQueueTask() < 0 )
        {
            CloseTask();
            Close();
            ret = -2;
            break;
        }

        {
            boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
            if(m_Reader.GetMsgQueueSize()>0)
            {
                m_pServer->AddTask(SharedFromThis());
            }
            else
            {
                CloseTask();
            }
        } 
#else
        while( true )
        {
            if( RecvQueueTask() < 0 )
            {
                CloseTask();
                Close();
                ret = -2;
                break;
            }

            task_time = get_current_tick() - start_tick;
            if( task_time > 500 )
            {
                CloseTask();
                break;
            }

            {
                boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
                if(m_Reader.GetMsgQueueSize() == 0)
                {
                    CloseTask();
                    break;
                }
            }
        }
#endif
    } while (0);

    if (ret < 0)
    {
        ERROR_LOG("Session(%p), ret=%d", this, ret);
    }
    else if( task_time > 300 )
    {
        ERROR_LOG("Session(%p)-->(%u),end...", this, task_time);
    }

    return ret;
}

int CAYSession::RecvQueueTask()
{
    int ret = 0;
    do 
    {
        SDataBuff req_msg;
        {
            boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
            if ( m_Reader.PopMsg(req_msg) < 0 )
            {
                ret = -1;
                break;
            }
        }
        
        if(m_ExchgKey.IsExchangeKey())
        {
            if( m_ExchgKey.DecryMsg( (uint8*)req_msg.get_buffer(), req_msg.data_size()) < 0 )
            {
                ret = -3;
                break;
            }
        }

        CDataStream t_ds(req_msg.get_buffer(), req_msg.data_size());
        protocol::MSG_HEADER header;
        t_ds >> header;

        CDataStream req_ds(req_msg.get_buffer(), req_msg.data_size());

        SDataBuff resp_msg(MAX_AY_MSG_SIZE);
        CDataStream resp_ds(resp_msg.get_buffer(), resp_msg.buffer_size());

        if( header.msg_id == protocol::MSG_ID_EXCHANGE_KEY )
        {
            if( HandleExchangeKey(req_ds, resp_ds) < 0 )
            {
                ret = -2;
                break;
            }
            resp_msg.data_size_ = resp_ds.size();

            //send exchange key response, not use encry
            if( SendMsg(resp_msg) < 0 )
            {
                ret = -5;
                break;
            }
        }
        else
        {
            CHostInfo hiRemote(GetRemoteIP(), GetRemotePort());
            if( m_pServer->OnTCPMessage(SharedFromThis(), hiRemote, header.msg_id, req_ds, resp_ds) < 0 )
            {
                ret = -4;
                break;
            }
            
            if( resp_ds.size() >0 )
            {
                resp_msg.data_size_ = resp_ds.size();
                if ( SendMsg(resp_msg) < 0 )
                {
                    ret = -6;
                    break;
                }
            }
        }        

    } while (0);

    if(ret<0)
    {
        ERROR_LOG("Session(%p), ret=%d", this, ret);
    }   

    return ret;
}
/*
int CAYSession::OpenTask()
{
    TRACE_LOG("Session(%p), task_flag=%d", this, m_bInTasking);

    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( !m_bInTasking )
    {
        m_bInTasking = true;
        m_pServer->AddTask(SharedFromThis());
    }

    return 0;
}*/

int CAYSession::CloseTask()
{
    TRACE_LOG("Session(%p), task_flag=%d", this, m_bInTasking);
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    m_bInTasking = false;
    return 0;
}

int CAYSession::OnRecv( const void* data, size_t data_len )
{
    TRACE_LOG( "Session(%p), recv data_len(%u).", this, data_len );
    int ret = 0;
    do 
    {
        if(!m_pServer)
        {
            break;
        }

        {
            boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
            ret = m_Reader.Push(data, data_len);
            if(ret<0)
            {
                ERROR_LOG("Session(%p), read msg failed, ret=%d", this, ret);
                break;
            }

            int recv_msg_que_size = m_Reader.GetMsgQueueSize();
            if(recv_msg_que_size>20)
            {
                WARN_LOG( "Session(%p), recv msg(%d), recv queue size(%u).", 
                    this, data_len, recv_msg_que_size );
            }
            else
            {
                TRACE_LOG( "Session(%p), recv msg(%d), recv queue size(%u).", 
                    this, data_len, recv_msg_que_size );
            }

            if( !m_bInTasking && recv_msg_que_size )
            {
                m_bInTasking = true;
                m_pServer->AddTask(SharedFromThis());            
                TRACE_LOG("Session(%p), task_flag=%d", this, m_bInTasking);
            }
        }

    } while (0);

    return ret;
}

int CAYSession::OnWrite( size_t data_len )
{
    return 0;
}

int CAYSession::OnClosed()
{
    DEBUG_LOG( "Session(%p), closed.", this );

    CHostInfo hiRemote(GetRemoteIP(), GetRemotePort());
    
    return m_pServer->OnTCPClosed(SharedFromThis(), hiRemote);
}

int CAYSession::HandleExchangeKey(IN CDataStream& recvds, OUT CDataStream& sendds)
{
    do 
    {
        protocol::MSG_HEADER header;
        protocol::ExchangeKeyRequest req;
        recvds >> header;
        recvds >> req;

        DEBUG_LOG("Session(%p), key_P_len=%u, key_A_len=%u, except_algorithm=%d!", this, req.key_P_length, req.key_A_length, req.except_algorithm);

        int ret = m_ExchgKey.OnExchangeKeyRequest(req);
        if( ret  < 0 )
        {
            ERROR_LOG("Session(%p), exchange key request error, ret=%d.", this, ret );
            break;
        }

        protocol::ExchangeKeyResponse resp;
        ret = m_ExchgKey.BuildExchangeKeyResponse(resp);
        if( ret  < 0 )
        {
            ERROR_LOG("Session(%p), bulid exchange key response error, ret=%d.", this, ret );
            break;
        }

        protocol::MsgHeader msg_head;
        msg_head.msg_id = protocol::MSG_ID_EXCHANGE_KEY;
        msg_head.msg_type = protocol::MSG_TYPE_RESP;
        msg_head.msg_seq = header.msg_seq;

        sendds << msg_head;
        sendds << resp;
        *(uint16*)sendds.getbuffer() = sendds.size();
        DEBUG_LOG("Session(%p), exchange key response, resp_code=%u,encry_algorithm=%d!", this, resp.resp_code, resp.encry_algorithm);
        return 0;
    } while (0);        
    return -1;
}

int CAYSession::SendFunc( SDataBuff& data_buff )
{
    if(m_ExchgKey.IsExchangeKey())
    {
        if( m_ExchgKey.EncryMsg( (uint8*)data_buff.get_buffer(), data_buff.data_size()) < 0 )
        {
            ERROR_LOG("Session(%p), encry msg failed.", this );
            return -1;
        }
    }

    return SendMsg(data_buff);
}

int CAYSession::SendFunc( uint8 * data, uint32 data_len )
{
    if(m_ExchgKey.IsExchangeKey())
    {
        if( m_ExchgKey.EncryMsg( data, data_len) < 0 )
        {
            ERROR_LOG("Session(%p), encry msg failed.", this );
            return -1;
        }
    }
    return SendMsg(data, data_len);
}

uint32 CAYSession::GetSendQLengthFunc()
{
    return GetSendQueueSize();
}

uint32 CAYSession::GetSendSpeed( unsigned int recent_second )
{
    return 0;
}

uint32 CAYSession::GetRecvSpeed( unsigned int recent_second )
{
    return 0;
}

CHostInfo CAYSession::GetLocalHost()
{
    CHostInfo hiLocal(GetRemoteIP(), GetRemotePort());

    return hiLocal;
}

CHostInfo CAYSession::GetRemoteHost()
{
    CHostInfo hiRemote(GetLocalIP(), GetLocalPort());

    return hiRemote;
}

std::ostringstream& CAYSession::DumpInfo( std::ostringstream& oss )
{
    return oss;
}