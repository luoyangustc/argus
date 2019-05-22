#ifndef __ITCPCLIENTSOCKET_H__
#define __ITCPCLIENTSOCKET_H__

#include <list>
#include <boost/shared_ptr.hpp>
#include "IClientSocketSink.h"

class ITCPClient
{
public:
	virtual int AdviseSink(ITCPClientSink * sink) = 0;
	virtual int UnadviseSink() = 0;
	virtual int Connect(uint32 ip, uint16 port) = 0;
	virtual int Send(const uint8* data_buff, uint32 data_len) = 0;
	virtual int Close() = 0;
	virtual bool CanSend() = 0;
};

typedef boost::shared_ptr<ITCPClient>         ITCPClient_ptr;
typedef std::list<ITCPClient_ptr>             ITCPClientList;
typedef std::list<ITCPClient_ptr>::iterator   ITCPClientListIter;

extern "C" ITCPClient* CreateAYTCPClient();
extern "C" void DestroyAYTCPClient(ITCPClient* pClient);

#endif //__ITCPCLIENTSOCKET_H__
