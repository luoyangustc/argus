#ifndef __USER_MGR_H__
#define __USER_MGR_H__

#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include "protocol/include/protocol_header.h"
#include "protocol/include/protocol_client.h"
#include "base/include/HostInfo.h"
#include "base/include/DeviceID.h"
#include "UserContext.h"
#include "WebUserContext.h"

using namespace std;
using namespace protocol;

class CUserMgr
{
public:
	CUserMgr(); 
	~CUserMgr(void);
public:
	void Update();
	void DoIdleTask();
	bool OnTCPClosed( const CHostInfo& hiRemote );
    ostringstream& DumpInfo( ostringstream& oss );
    ostringstream& DumpUserInfo( const string& user_name, ostringstream& oss );
public:
    string GetUserName(const CHostInfo& hiRemote);
    uint32 GetConnectNum();
    CUserContext_ptr GetUserContext( const CHostInfo& hiRemote );
    void GetUserContextsByUsername( const string& user_name, OUT vector<CUserContext_ptr>& user_contexts );
    void RemoveUserContext(const CHostInfo& hiRemote);
public:
    //client message handles
    bool ON_CuLoginRequest( ITCPSessionSendSink* sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuLoginReq& req, CuLoginResp& resp );
    bool ON_CuStatusReport( ITCPSessionSendSink* sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuStatusReportReq& req, CuStatusReportResp& resp );
    bool ON_CuMediaOpenRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaOpenReq& req );
    bool ON_CuMediaCloseRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, uint32 msg_seq, const CuMediaCloseReq& req,  CuMediaCloseResp& resp );
public:
    bool OnWebUserRequest( ITCPSessionSendSink*sink, const CHostInfo& hiRemote, SHttpRequestPara_ptr pReq );
private:
	boost::recursive_mutex lock_;
    map<CHostInfo, CUserContext_ptr> user_contexts_;
    map<CHostInfo, CWebUserContext_ptr> webcontexts_;
};

typedef boost::shared_ptr<CUserMgr> CUserMgr_ptr;

#endif //__C3_USER_MGR_H__
