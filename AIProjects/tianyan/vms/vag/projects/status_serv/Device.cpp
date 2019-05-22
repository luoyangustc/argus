#include "Device.h"
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
#include "base/include/to_string_util.h"
#include "base/include/variant.h"
#include "base/include/httpdownloader.h"
#include "base/include/logging_posix.h"
#include "third_party/json/include/reader.h"
#include "third_party/json/include/json.h"

Device::Device()
    : device_id_()
    , status_(kNormalOffline)
    , session_server_addr_()
    , channel_list_size_(0)
{
}

Device::~Device()
{

}

bool Device::OnStatusReport(const protocol::SDeviceSessionStatus& device_status)
{
    if( device_status.mask & 0x01 )
    {
        device_id_ = device_status.did;
    }
    else
    {
        Error("device session status report msg error, (0x%x)\n", device_status.mask );
        return false;
    }

    if( device_status.mask & 0x02 )
    {
        status_ = device_status.status;
        timestamp_ = device_status.timestamp;
        Debug("device status-->(%s, %s, %llu.%llu)\n",
            device_id_.c_str(),
            protocol::SDeviceSessionStatus::enm_dev_status_online==device_status.status ? "online":"offline",
            timestamp_.tv_sec, timestamp_.tv_usec );
    }

    if( device_status.mask & 0x04 )
    {
        version_ = device_status.version;
        dev_type_ = device_status.dev_type;
        channel_num_ = device_status.channel_num;
        session_server_addr_ = CHostInfo(device_status.session_serv_addr.ip, device_status.session_serv_addr.port);

        Debug("device info-->(%s, %s, %d, %d, %s)\n",
            device_id_.c_str(),
            version_.c_str(),
            dev_type_,
            channel_num_,
            session_server_addr_.GetNodeString().c_str() );
    }

    if( device_status.mask & 0x08 )
    {
        channel_list_size_ = device_status.channel_list_size;
        channel_list_ = device_status.channel_list;

        {
            string channel_status = "";
            for(int i = 0; i < channel_num_; i++ )
            {
                channel_status += "0";
            }

            vector<protocol::DevChannelInfo>::iterator it = channel_list_.begin();
            for( ; it!=channel_list_.end(); ++it )
            {
                if( it->channel_status == protocol::CHANNEL_STS_ONLINE 
                    && it->channel_id <= channel_num_ )
                {
                    int pos = it->channel_id - 1;
                    channel_status.replace( pos, 1, 1, '1');
                }
            }
            Debug("device channel status-->(%s, %d, %s)\n", device_id_.c_str(), channel_list_size_, channel_status.c_str() );
        }
        
    }

    return true;
}

std::string Device::ToTimestamp(const protocol::STimeVal& tv)
{
    char buffer[64] = "";
    char* position = buffer;
    size_t len = strftime(position, sizeof buffer, "%Y-%m-%d %H:%M:%S",
        localtime((time_t*)&tv.tv_sec));
    if ( 0 == len || snprintf(position + len, sizeof(buffer) - len, ".%llu", tv.tv_usec) < 0)
    {
        return std::string();
    }
    else
    {
        return buffer;
    }
}

inline std::string Device::ToString(int int_to_string)
{
    char buffer[64] = "";
    snprintf(buffer,sizeof buffer,"%d",int_to_string);
    return buffer;
}