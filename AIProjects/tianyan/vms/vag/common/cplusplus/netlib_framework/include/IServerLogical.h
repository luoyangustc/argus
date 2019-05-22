
#ifndef __ISERVER_LOGICAL_H__
#define __ISERVER_LOGICAL_H__

#include <string>
using namespace std;
#include "typedefine.h"
#include "typedef_win.h"
#include "datastream.h"
#include "HostInfo.h"
#include "ITCPSessionSendSink.h"

/***********************************************************
*  消息类型定义
*enum MSG_TYPE
*{
*    MSG_TYPE_EXCHANGE_REQ   = 0x00000001,   //密钥交换请求
*    MSG_TYPE_EXCHANGE_RESP  = 0x00000002,   //密钥交换响应
*    MSG_TYPE_CMD_REQ        = 0x00000003,   //Command请求
*    MSG_TYPE_CMD_RESP       = 0x00000004,   //Command响应
*    MSG_TYPE_ECHO_TEST      = 0x00000005,   //反射测试消息
*};
*************************************************************/

class IServerLogical
{
public:
    IServerLogical(){}
    virtual ~IServerLogical(){}
public:
    virtual int32 OnHttpClientRequest(ITCPSessionSendSink*sink,CHostInfo& hiRemote,SHttpRequestPara_ptr pReq,SHttpResponsePara_ptr pRes){return -1;}
    virtual int32 OnUDPMessage(CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds, IN int thread_index,uint8 algo){return -1;}
    virtual int32 OnTCPMessage(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_type, CDataStream& recvds,CDataStream& sendds){return -1;}
    virtual int32 OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& hiRemote,CDataStream& sendds){return -1;}
    virtual int32 OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& hiRemote){return -1;}

    virtual bool Start(){return false;}
    virtual void Stop(){}

    virtual void Update(){}
    virtual void DoIdleTask(){}
public:
    static IServerLogical * GetInstance();
};

#endif //__ISERVER_LOGICAL_H__

