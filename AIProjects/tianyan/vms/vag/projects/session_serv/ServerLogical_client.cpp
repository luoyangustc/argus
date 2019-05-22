
#include "logging_posix.h"
#include "ServerLogical.h"

bool CServerLogical::ON_CuLoginRequest(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds)
{
    do
    {
        CuLoginReq req;
        recvds >> req;
        if(!recvds.good_bit())
        {
            Error("from(%s), parse msg error!",
                hiRemote.GetNodeString().c_str() );
            break;
        }

        if ( !(req.mask&0x01) || !(req.mask&0x02) )
        {
           Warn("from(%s), mask error, mask=0x%x!",
                hiRemote.GetNodeString().c_str(), req.mask);
            break;
        }

        if ( req.user_name.empty() )
        {
           Warn("from(%s), user name is empty!",
                hiRemote.GetNodeString().c_str());
            break;
        }

        Debug("from(%s), user_name:%s, private ip:%s", 
            hiRemote.GetNodeString().c_str(), req.user_name.c_str(), req.private_ip.c_str());

        CuLoginResp resp;
        if ( !pUserMgr_->ON_CuLoginRequest( sink, hiRemote, msg_seq, req, resp ) )
        {
            break;
        }

        return true;

    }while(false);

    return false;
}

bool CServerLogical::ON_CuStatusReport(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds)
{
	do
	{
        CuStatusReportReq req;
        recvds >> req;
        if(!recvds.good_bit())
        {
           Warn("from(%s), Parse Message  Error!",
                hiRemote.GetNodeString().c_str() );
            break;
        }

        CuStatusReportResp resp;
        if( !pUserMgr_->ON_CuStatusReport(sink, hiRemote, msg_seq, req, resp) )
        {
            break;
        }

        return true;

	}while(false);

	return false;
}

bool CServerLogical::ON_CuMediaOpen(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds)
{
    do
    {
        CuMediaOpenReq req;
        recvds >> req;
        if(!recvds.good_bit())
        {
           Warn("from(%s), Parse Message  Error!",
                hiRemote.GetNodeString().c_str() );
            break;
        }

        if( !pUserMgr_->ON_CuMediaOpenRequest(sink, hiRemote, msg_seq, req))
        {
            break;
        }

        return true;
    }
    while(false);

    return false;
}

bool CServerLogical::ON_CuMediaClose(ITCPSessionSendSink* sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds)
{
    do
    {
        CuMediaCloseReq req;
        recvds >> req;
        if(!recvds.good_bit())
        {
           Warn("from(%s), Parse Message  Error!",
                hiRemote.GetNodeString().c_str() );
            break;
        }

        CuMediaCloseResp resp;
        if( !pUserMgr_->ON_CuMediaCloseRequest(sink, hiRemote, msg_seq, req, resp))
        {
            break;
        }

        return true;
    }
    while(false);

    return false;
}

