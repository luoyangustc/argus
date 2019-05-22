#include "HTTPSession.h"
#include "HostInfo.h"
#include "datastream.h"
#include "Log.h"

CHTTPSession::CHTTPSession(boost::asio::io_service& io_svc)
    :CTCPHandler(io_svc)
    ,m_RequestMsg(MAX_HTTP_REQUEST_MSG_SIZE)
{

}

CHTTPSession::~CHTTPSession()
{

}

int CHTTPSession::Init( CTCPServer<CHTTPSession>* pServ )
{
    do
    {
        if( !pServ )
        {
            ERROR_LOG( "Session(%p), Remote(%s), Server is nil!", this, GetRemoteAddr().c_str() );
            break;
        }

        if( !m_spHttpReqPara.get() )
        {
            m_spHttpReqPara.reset( new SHttpRequestPara() );
            if( !m_spHttpReqPara.get() )
            {
                ERROR_LOG( "Session(%p), Remote(%s), Create http request para failed!", this, GetRemoteAddr().c_str() );
                break;
            }
        }

        if( !m_spHttpRespPara.get() )
        {
            m_spHttpRespPara.reset( new SHttpResponsePara() );
            if( !m_spHttpRespPara.get() )
            {
                ERROR_LOG( "Session(%p), Remote(%s), Create http response para failed!", this, GetRemoteAddr().c_str() );
                break;
            }            
        }
        
        m_pServer = pServ;  

        //Set tcp aysnc mode
        SetMode(en_tcp_on_async);

        //Open tcp handle
        if( Open() < 0 )
        {
            ERROR_LOG( "Session(%p), Remote(%s), start failed!", this, GetRemoteAddr().c_str() );
            break;
        }

        DEBUG_LOG( "Session(%p), Remote(%s), Parser(%p),Init success.", this, GetRemoteAddr().c_str(), &m_ReqParser );
        return 0;

    } while (0);

    return -1;
}

void CHTTPSession::Reset()
{
    CTCPHandler::Reset();
    m_ReqParser.Reset();
    m_Response.Reset();
    m_pServer = NULL;
    m_RequestMsg.clear();
}

int CHTTPSession::TaskMain()
{
    int ret = 0;
    do
    {
        // ¢Ù handle http request
        CHostInfo hiRemote(GetRemoteIP(), GetRemotePort());
        if( m_pServer->OnHttpClientRequest( SharedFromThis(), hiRemote, m_spHttpReqPara, m_spHttpRespPara ) < 0 )
        {
            ret = -1;
            break;
        }

        // ¢Ú do http response
        if(!m_spHttpRespPara->ret_code.empty())
        {
            if( !m_Response.Set( m_spHttpRespPara ) )
            {
                ret = -2;
                break;
            }

            if( SendMsg( m_Response.ToBuffers() ) < 0 )
            {
                ret = -3;
                break;
            }
        }
    } while (0);

    if(ret <0)
    {
        Close();
        ERROR_LOG("Session(%p), task end, ret=%d", this, ret);
    }
    return ret;
}

boost::shared_ptr<CHTTPSession> CHTTPSession::SharedFromThis()
{
    return boost::dynamic_pointer_cast<CHTTPSession>(shared_from_this());
}

int CHTTPSession::OnRecv(const void* data, size_t data_len)
{
    do 
    {
        if( (m_RequestMsg.data_size()+data_len) > m_RequestMsg.buffer_size() )
        {
            ERROR_LOG( "Session(%p), Remote(%s), The received data exceeds the length of the buffer!", this, GetRemoteAddr().c_str() );
            Close();
            break;
        }

        if( !m_RequestMsg.push_back(data, data_len) )
        {
            ERROR_LOG( "Session(%p), Remote(%s), receive request msg faile!", this, GetRemoteAddr().c_str() );
            Close();
            break;
        }

        EN_PARSER_RESULT result = m_ReqParser.ParserExe(m_RequestMsg.get_buffer(), m_RequestMsg.data_size(), 
                                       m_spHttpReqPara->header_detail,
                                       m_spHttpReqPara->header_data,
                                       m_spHttpReqPara->header_data_len,
                                       m_spHttpReqPara->content_data,
                                       m_spHttpReqPara->content_data_len);
        if( result == EN_PARSER_RST_OK )
        {
            DEBUG_LOG("Session(%p), add task...", this);
            m_pServer->AddTask( SharedFromThis() );
        }
        else if(  result == EN_PARSER_RST_NO_COMPLETE  )
        {
            WARN_LOG( "Session(%p), Remote(%s), recv_len(%u),msg is not complete!", this, GetRemoteAddr().c_str(), m_RequestMsg.data_size());
        }
        else
        {
            ERROR_LOG( "Session(%p), Remote(%s), Read data is incorrect", this, GetRemoteAddr().c_str() );
            //Close(); //Read error, stop!
            break;
        }

        return 0;
    } while (0);
    return -1;
}

int CHTTPSession::OnWrite( size_t data_len )
{
    TRACE_LOG( "Session(%p), Remote(%s), written(%u).", this, GetRemoteAddr().c_str(), data_len );
    Close();
    return 0;
}

int CHTTPSession::OnClosed()
{
    //CHostInfo hiRemote(GetRemoteIP(), GetRemotePort());
    //m_pServer->OnTCPClosed(SharedFromThis(), hiRemote);
    DEBUG_LOG( "Session(%p), Remote(%s).", this, GetRemoteAddr().c_str() );
    return 0;
}

int CHTTPSession::SendHttpResp( SHttpResponsePara_ptr pResp )
{
    if( !m_Response.Set( pResp ) )
    {
        return -1;
    }

    if( SendMsg( m_Response.ToBuffers() ) < 0 )
    {
        return -2;
    }

    return 0;
}

int CHTTPSession::SendFunc( SDataBuff& data_buff )
{
    return SendMsg(data_buff);
}

int CHTTPSession::SendFunc( uint8 * data, uint32 data_len )
{
    return SendMsg(data, data_len);
}

uint32 CHTTPSession::GetSendQLengthFunc()
{
    return 0;
}

uint32 CHTTPSession::GetSendSpeed( unsigned int recent_second )
{
    return 0;
}

uint32 CHTTPSession::GetRecvSpeed( unsigned int recent_second )
{
    return 0;
}

CHostInfo CHTTPSession::GetLocalHost()
{
    CHostInfo hiLocal(GetRemoteIP(), GetRemotePort());

    return hiLocal;
}

CHostInfo CHTTPSession::GetRemoteHost()
{
    CHostInfo hiRemote(GetLocalIP(), GetLocalPort());

    return hiRemote;
}

std::ostringstream& CHTTPSession::DumpInfo( std::ostringstream& oss )
{
    return oss;
}

