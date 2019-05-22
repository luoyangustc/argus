#ifndef __STREAM_H__
#define __STREAM_H__

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "base/include/HostInfo.h"
#include "protocol/include/protocol_status.h"

class Stream : boost::noncopyable
{
public:
    Stream();
    ~Stream();
    bool OnStatusReport(const protocol::SDeviceStreamStatus& status);
public:
    string session_id_;
    uint16 session_type_;
    string device_id_;
    uint16 channel_id_;
    uint16 stream_id_;
    uint8 status_;
    protocol::STimeVal timestamp_;
    CHostInfo stream_serv_addr_;
};

typedef boost::shared_ptr<Stream> StreamPtr;

#endif  // __STREAM_H__
