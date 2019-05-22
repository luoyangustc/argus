#ifndef STATUS_SERVER_STATUSSERVER_H
#define STATUS_SERVER_STATUSSERVER_H

#include <string>
#include <map>
#include <boost/noncopyable.hpp>
#include "base/include/typedefine.h"
#include "base/include/HostInfo.h"
#include "base/include/datastream.h"
#include "protocol/include/protocol_header.h"
#include "netlib_framework/include/IServerLogical.h"
#include "netlib_framework/include/ITCPSessionSendSink.h"
#include "HttpRequestHandler.h"
#include "SessionMgr.h"
#include "DeviceMgr.h"
#include "StreamMgr.h"

using namespace std;

class StatusServer : public IServerLogical, boost::noncopyable
{
public: 
  StatusServer();
  virtual ~StatusServer();
  
  virtual bool Start(uint16 http_port,uint16 server_port);
  virtual void Stop();
  virtual void Update();
  virtual void DoIdleTask();

  virtual int32 OnTCPMessage(ITCPSessionSendSink*sink, CHostInfo& peer_addr, uint32 msg_id, CDataStream& recvds, CDataStream& sendds);
  virtual int32 OnTCPAccepted(ITCPSessionSendSink*sink, CHostInfo& peer_addr, CDataStream& sendds);
  virtual int32 OnTCPClosed(ITCPSessionSendSink*sink, CHostInfo& peer_addr);
  virtual int32 OnHttpClientRequest(ITCPSessionSendSink*sink, CHostInfo& peer_addr, SHttpRequestPara_ptr http_request, SHttpResponsePara_ptr http_resp);
public:
    static StatusServer* ServerInstance();
    DeviceMgrPtr SessionDevMgrInstance(){return device_mgr_;}
    StreamMgrPtr StreamDevMgrInstance(){return stream_mgr_;}
private:
  //stream/session server message
  bool OnLogin(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds);
  bool OnLoadReport(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds);
  bool OnStatusReport(ITCPSessionSendSink*sink, const CHostInfo& peer_addr, CDataStream& recvds, CDataStream& sendds);

  SessionMgrPtr session_mgr_;
  DeviceMgrPtr device_mgr_;
  StreamMgrPtr stream_mgr_;
  HttpRequestHandlerPtr http_request_handler_;
  uint16 http_port_;
  uint16 serv_port_;
};

StatusServer* GetService();
DeviceMgrPtr GetDeviceMgr();
StreamMgrPtr GetStreamMgr();
std::string TimestampStr(const protocol::STimeVal& tv);


#endif  // STATUS_SERVER_STATUSSERVER_H
