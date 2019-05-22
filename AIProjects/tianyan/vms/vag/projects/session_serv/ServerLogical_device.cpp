#include <boost/thread.hpp>
#include "logging_posix.h"
#include "tick.h"
#include "ServerLogical.h"

bool CServerLogical::ON_DeviceLoginRequest(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds)
{
	do
	{
		DeviceLoginReq req;
		recvds >> req;
		if (!recvds.good_bit())
		{
			Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
			break;
		}

		DeviceLoginResp resp;

		return pDeviceMgr_->ON_DeviceLoginRequest(sink,hiRemote,msg_seq,req,resp);

	} while(false);

	return false;
}

bool CServerLogical::ON_DeviceAbilityReport(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds)
{
	do 
	{
		DeviceAbilityReportReq req;
		recvds >> req;
		if (!recvds.good_bit())
		{
			Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
			break;
		}

		return pDeviceMgr_->ON_DeviceAbilityReport(sink,hiRemote,msg_seq,req);
	} while(false);
	return false;
}

bool CServerLogical::ON_StatusReport(ITCPSessionSendSink*sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds,CDataStream& sendds)
{
	do
	{
		DeviceStatusReportReq req;
		recvds >> req;
		if (!recvds.good_bit())
		{
			Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
			break;
		}
		DeviceStatusReportResp resp;
		if( !pDeviceMgr_->ON_StatusReport(sink, hiRemote, msg_seq, req, resp) )
		{
			break;
		}

		return true;

	} while(false);

	return false;
}

bool CServerLogical::ON_DeviceAlarmReport(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds)
{
    do
    {
        DeviceAlarmReportReq req;
        recvds >> req;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }
        DeviceAlarmReportResp resp;
        if( !pDeviceMgr_->ON_DeviceAlarmReport(sink, hiRemote, msg_seq, req, resp) )
        {
            break;
        }

        return true;

    } while(false);

    return false;
}

bool CServerLogical::ON_DeviceMediaOpenResp(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_seq,CDataStream& recvds,CDataStream& sendds)
{
    do 
    {
        DeviceMediaOpenResp resp;
        recvds >> resp;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }
        
        if( !pDeviceMgr_->ON_DeviceMediaOpenResp(sink, hiRemote, msg_seq, resp) )
        {
            break;
        }

        return true;

    } while(false);

    return false;
}

bool CServerLogical::ON_DeviceSnapResp(ITCPSessionSendSink*sink, CHostInfo& hiRemote, uint32 msg_seq, CDataStream& recvds, CDataStream& sendds)
{
    do 
    {
        DeviceSnapResp resp;
        recvds >> resp;
        if (!recvds.good_bit())
        {
            Error("from(%s), parse msg error!", hiRemote.GetNodeString().c_str());
            break;
        }
        
        if( !pDeviceMgr_->ON_DeviceSnapResp(sink, hiRemote, msg_seq, resp) )
        {
            break;
        }

        return true;

    } while(false);

    return false;
}
