#include <sys/time.h>
#include "DeviceMgr.h"
#include <boost/thread/lock_guard.hpp>
#include "base/include/datastream.h"
#include "base/include/logging_posix.h"

DeviceMgr::DeviceMgr()
{

}

DeviceMgr::~DeviceMgr()
{

}

void DeviceMgr::Update()
{

}

DevicePtr DeviceMgr::GetDevice(const string& device_id)
{
    DevicePtr pDevice;
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        DeviceIterator it = all_devices_.find(device_id);
        if( it!=all_devices_.end() )
        {
            pDevice = it->second;
        }
    }
    return pDevice;
}

void DeviceMgr::GetOnlineDeivce(vector<string>& device_list)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    DeviceIterator it = all_devices_.begin();
    for( ; it != all_devices_.end(); ++it )
    {
        if( it->second->status_ == protocol::SDeviceSessionStatus::enm_dev_status_online )
        {
            device_list.push_back(it->second->device_id_);
        }
    }
}

void DeviceMgr::GetOfflineDeivce(vector<string>& device_list)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    DeviceIterator it = all_devices_.begin();
    for( ; it != all_devices_.end(); ++it )
    {
        if( it->second->status_ != protocol::SDeviceSessionStatus::enm_dev_status_online )
        {
            device_list.push_back(it->second->device_id_);
        }
    }
}

int DeviceMgr::OnStatusReport(CDataStream& recvds, CDataStream& sendds)
{
    protocol::MsgHeader header;
    recvds >> header;
    if ( !recvds.good_bit() )
    {
        Error("Parse MsgHeader Message Error!\n");
        return -1;
    }

    protocol::StsSessionStatusReportReq req;
    recvds >> req;
    if ( !recvds.good_bit() )
    {
        Error("Parse StsSessionStatusReportReq Message Error!\n");
        return -2;
    }
    
    protocol::StsSessionStatusReportResp resp;
    resp.mask = 0x01;
    if ( !HandleEveryDeviceStatus(req) )
    {
        resp.resp_code = -10001;  // FIXME : error code ?       
    }
    else
    {
        resp.resp_code = protocol::EN_SUCCESS;     
    }

    header.msg_id = protocol::MSG_ID_STS_SESSION_STATUS_REPORT;
    header.msg_type = protocol::MSG_TYPE_RESP;
    sendds << header;
    sendds << resp;
    *((WORD*)sendds.getbuffer()) = sendds.size();       
    return 0;
}

int DeviceMgr::OnSessionOffline(const CHostInfo& hi_remote)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    std::map<CHostInfo, set<string> >::iterator it = session_devices_.find(hi_remote);
    if( it!=session_devices_.end() )
    {
        struct timeval cur_tv;
        gettimeofday(&cur_tv, NULL);

        set<string>& devices = it->second;
        set<string>::iterator itDevId = devices.begin();
        for( ; itDevId!=devices.end(); ++itDevId)
        {
            DeviceIterator itDev = all_devices_.find(*itDevId);
            if( itDev != all_devices_.end() )
            {
                itDev->second->status_ = Device::kAbnormalOffline;
                itDev->second->timestamp_.tv_sec = (uint64)cur_tv.tv_sec;
                itDev->second->timestamp_.tv_usec = (uint64)cur_tv.tv_usec;
            }
            
        }
        session_devices_.erase(it);
    }

    return 0;
}

bool DeviceMgr::HandleEveryDeviceStatus( protocol::StsSessionStatusReportReq& req )
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    vector<protocol::SDeviceSessionStatus>::iterator it = req.devices.begin();
    for ( /*void*/; it != req.devices.end(); ++it )
    {
        const protocol::SDeviceSessionStatus& session_status = *it;
        const std::string device_id = session_status.did;

        bool is_new_dev = false;
        DevicePtr pDevice;
        DeviceIterator itDev = all_devices_.find(device_id);
        if ( itDev == all_devices_.end() )  // not found
        {
            is_new_dev = true;
            pDevice.reset(new Device());

            if ( !pDevice || !pDevice->OnStatusReport(session_status))
            {
                return false;
            }
        }
        else
        {
            pDevice = itDev->second;
            if ( !pDevice->OnStatusReport(session_status) )
            {
                return false;
            }
        }

        if( session_status.mask & 0x02 )
        {
            CHostInfo hiSession( session_status.session_serv_addr.ip.c_str(), session_status.session_serv_addr.port );
            if( is_new_dev && session_status.status == protocol::SDeviceSessionStatus::enm_dev_status_online )
            {
                all_devices_[device_id] = pDevice;
            }

            if( session_status.status == protocol::SDeviceSessionStatus::enm_dev_status_online )
            {
                if(session_devices_[hiSession].find(session_status.did) == session_devices_[hiSession].end() )
                {
                    session_devices_[hiSession].insert(session_status.did);
                }
            }
            else if ( session_status.status == protocol::SDeviceSessionStatus::enm_dev_status_offline )
            {
                std::map<CHostInfo, set<string> >::iterator it = session_devices_.find(pDevice->session_server_addr_);
                if( it!=session_devices_.end() )
                {
                    it->second.erase(pDevice->device_id_);
                }
            }
        }
    }

    return true;
}
