
#ifndef __SERVER_LOGICAL_H__
#define __SERVER_LOGICAL_H__

#include <boost/shared_array.hpp>
#include <set>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "server/IServerLogical.h"
#include "typedefine.h"
#include "tick.h"
#include "datastream.h"
#include "HostInfo.h"

using namespace std;

class CServerLogical : public IServerLogical
{
public:
    CServerLogical();
    ~CServerLogical();
public:
    virtual bool Start();
    virtual void Stop();

    virtual int32 OnHttpClientRequest(CHostInfo& hiRemote,SHttpRequestPara_ptr pReq,SHttpResponsePara_ptr pRes);
    virtual int32 OnUDPMessage(CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds, IN int thread_index,uint8 algo);
    virtual int32 OnTCPMessage(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_type, CDataStream& recvds,CDataStream& sendds);
    virtual int32 OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& hiRemote,CDataStream& sendds);
    virtual int32 OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& hiRemote);

    virtual void Update();
    virtual void DoIdleTask();

public:
	ostringstream& DumpInfo(ostringstream& oss,const string&type);
private:
	tick_t start_tick_;
private:
    static CServerLogical logic_;
public:
    static CServerLogical* GetLogical(){return &logic_;}

};

#endif //__SERVER_LOGICAL_H__

