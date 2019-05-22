#ifndef __DEVICE_MGR_H__
#define __DEVICE_MGR_H__

#include "IStatusMgr.h"
#include <string>
#include <map>
#include <set>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include "base/include/HostInfo.h"
#include "base/include/datastream.h"
#include "protocol/include/protocol_status.h"
#include "Device.h"

using namespace std;

class DeviceMgr : public IStatusMgr
{
public:
    typedef std::map<std::string, DevicePtr> DeviceMap;
    typedef std::map<std::string, DevicePtr>::iterator DeviceIterator;
    
    DeviceMgr();
    virtual ~DeviceMgr();
    void Update();
    virtual int OnStatusReport(CDataStream& recvds, CDataStream& sendds);
    virtual int OnSessionOffline(const CHostInfo& hi_remote);  // all device
    DevicePtr GetDevice(const string& device_id);
    void GetOnlineDeivce(vector<string>& device_list);
    void GetOfflineDeivce(vector<string>& device_list);
private:
    bool HandleEveryDeviceStatus(protocol::StsSessionStatusReportReq& req);
private:
    boost::recursive_mutex lock_;
    DeviceMap all_devices_;
    std::map<CHostInfo, set<string> > session_devices_;
};

typedef boost::shared_ptr<DeviceMgr> DeviceMgrPtr;

#endif  // __DEVICE_MGR_H__
