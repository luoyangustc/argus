#ifndef __HTTP_REQUEST_HANDLER__
#define __HTTP_REQUEST_HANDLER__

#include <string>
#include <map>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "base/include/typedefine.h"
#include "SessionMgr.h"

class CHostInfo;
class Variant;

enum EnHttpErrorCode
{
    kErrorNone = 0,
    kInvalidParameter=2000,
    kServerInternalError,
    kInvalidDeviceId,
    kTokenInvalid,
    kDeviceOffline
};

class HttpRequestHandler : boost::noncopyable
{
    typedef std::map<std::string,std::string> HttpParameterMap;
    typedef std::map<std::string,std::string>::iterator HttpParameterIterator;
public:
    explicit HttpRequestHandler(const SessionMgrPtr& session_mgr);
    ~HttpRequestHandler();

    int OnHttpRequest(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
private:
    int OnRoot(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnGenerateDeviceId(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnGetAccessToken(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnGetAccessServer(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnGetSessionServer(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnStreamRRS(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnQuerySessionServerInfo(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnQueryStreamServerInfo(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnQueryDeviceStatus(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnReqLive(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnDeviceControl(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
    int OnDeviceSnap(const CHostInfo& peer_addr,const SHttpRequestPara_ptr& req, SHttpResponsePara_ptr& resp);
private:

    bool QueryDeviceStatus(const std::string& device_id, Variant& reply);
    void MakeHttpResponse(const std::string& content, SHttpResponsePara_ptr& http_resp);
    int NotHandleOk(SHttpResponsePara_ptr& http_resp);  
    int HttpNotFound(SHttpResponsePara_ptr& http_resp);  

private:
    const SessionMgrPtr& session_mgr_;
};

typedef boost::shared_ptr<HttpRequestHandler> HttpRequestHandlerPtr;

#endif  // __HTTP_REQUEST_HANDLER__
