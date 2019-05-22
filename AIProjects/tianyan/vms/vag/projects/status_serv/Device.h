#ifndef STATUS_SERVER_DEVICE_H
#define STATUS_SERVER_DEVICE_H

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "base/include/typedefine.h"
#include "base/include/HostInfo.h"
#include "protocol/include/protocol_status.h"
using namespace std;
using namespace protocol;

class Device : boost::noncopyable
{
public:
    enum DeviceOnlineStatus { kNormalOffline, kAbnormalOffline };
    explicit Device();
    ~Device();  // This class will not be inherited

    bool OnStatusReport(const protocol::SDeviceSessionStatus&  status);
    std::string ToTimestamp(const protocol::STimeVal& tv);
    std::string ToString(int int_to_string);
public:    
    string device_id_;
    uint8 status_;
    protocol::STimeVal timestamp_;

    string version_;
    uint8 dev_type_;
    uint16 channel_num_;
    CHostInfo session_server_addr_;  // session server addr[ip:port]
    
    uint16 channel_list_size_;               // 通道数
    vector<DevChannelInfo> channel_list_;    // 通道状态列表, 可变长
};

typedef boost::shared_ptr<Device> DevicePtr;

#endif  // STATUS_SERVER_DEVICE_H
