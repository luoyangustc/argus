
#ifndef __ITCPSESSION_SENDSINK_H__
#define __ITCPSESSION_SENDSINK_H__
#include <fstream>
#include <sstream>
#include <iomanip>
#include "typedefine.h"
#include "HostInfo.h"
#include "BufferInfo.h"
using namespace std;

class ITCPSessionSendSink
{
public:
    virtual CHostInfo GetLocalHost() = 0;
    virtual CHostInfo GetRemoteHost() = 0;
    virtual int SendFunc( SDataBuff& data_buff ){ return -1; }
	virtual int SendFunc(uint8 * data, uint32 data_len){ return -1; }
    virtual uint32 GetSendQLengthFunc(){ return 0; }
    virtual uint32 GetSendSpeed(uint32 recent_second){ return 0; }
    virtual uint32 GetRecvSpeed(uint32 recent_second){ return 0; }
    virtual std::ostringstream& DumpInfo(std::ostringstream& oss){ return oss; }
    virtual int SendHttpResp(SHttpResponsePara_ptr pResp){ return -1; }
};
#endif //__ITCPSESSION_SENDSINK_H__
