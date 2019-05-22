#include "Stream.h"
#include "base/include/logging_posix.h"

Stream::Stream()
    : session_id_()
    , session_type_(0)
    , device_id_()
    , channel_id_(0)
    , stream_id_(0)
    , status_(0)
{
    
}

Stream::~Stream()
{
    
}

bool Stream::OnStatusReport(const protocol::SDeviceStreamStatus& status)
{
    session_id_ = status.session_id;
    session_type_ = status.session_type;
    device_id_ = status.did;
    channel_id_ = status.channel_id;
    stream_id_ = status.stream_id;
    status_ = status.status;
    timestamp_ = status.timestamp;
    stream_serv_addr_ = CHostInfo(status.stream_serv_addr.ip, status.stream_serv_addr.port);
    return true;
}
