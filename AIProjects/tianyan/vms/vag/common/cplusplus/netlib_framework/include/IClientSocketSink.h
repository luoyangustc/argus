#ifndef __ICLIENTSOCKET_SINK_H__
#define __ICLIENTSOCKET_SINK_H__

#include "typedefine.h"

class ITCPClientSink
{
public:
    virtual ~ITCPClientSink(){}
public:
    virtual int OnTCPConnected(uint32 ip, uint16 port) = 0;
    virtual int OnTCPConnectFailed(uint32 ip, uint16 port) = 0;
    virtual int OnTCPClose(uint32 ip, uint16 port) = 0;
    virtual int OnTCPMessage(uint32 ip, uint16 port, uint8* data, uint32 data_len) = 0;
};

#endif //__ICLIENTSOCKET_SINK_H__

