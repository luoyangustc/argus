#ifndef __MEDIA_SESSION__
#define __MEDIA_SESSION__

#include <time.h>
#include <string>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>
#include "base/include/DeviceChannel.h"
#include "base/include/web_request.h"
#include "base/include/tick.h"
#include "base/include/HostInfo.h"
#include "protocol/include/protocol_device.h"

using namespace std;
using namespace protocol;

enum SessionType 
{
    en_session_type_live = 1, 
    en_session_type_playback
};

enum SessionState
{ 
    en_session_sts_init = 0,
    en_session_sts_get_stream_serv,
    en_session_sts_establishing,
    en_session_sts_active,
    en_session_sts_close,
};

typedef struct tagMediaDesc
{
	bool video_open;
	bool audio_open;
	uint32 begin_time;
	uint32 end_time;

public:
	tagMediaDesc(){memset(this, 0, sizeof(tagMediaDesc));}
}SMediaDesc, *SMediaDescPtr;

class IMediaSessionSink
{
public:
    virtual void OnMediaOpenAck(int err_code, const SDeviceChannel& dc, SessionType session_type, const string& session_id, const CHostInfo& hi_stream_serv) = 0;
};

typedef boost::shared_ptr<IMediaSessionSink> IMediaSessionSink_ptr;

class MediaSession
{
    friend class CMediaSessionMgr;
public:
    MediaSession(const SDeviceChannel& dc, SessionType session_type, const string& session_id="");
    virtual ~MediaSession();

    bool IsAlive();
    void Stop();
    void Update();
public:
    bool OnUserMediaOpenReq(const CHostInfo& hiRemote, IMediaSessionSink_ptr user_ctx);
    bool OnUserClose(const CHostInfo& hiRemote);
    bool OnDeviceClose();
public:
    void UpdateSessionState( SessionState state );
    string GetStreamSchedulerUrl(bool is_relay, CHostInfo client_hi);

public:
    bool OnDeviceStatusReport(const protocol::DeviceMediaSessionStatus& status);
    bool OnDeviceMediaOpenAck(const DeviceMediaOpenResp& resp);
private:
    bool Start();
    void GenerateSessionId();
    static int OnHttpStreamScheduleCallback(uint32_t request_id, void* user_data, int err_code, int http_code, const char* http_resp);
    static bool ParseStreamHostaddr(const char* szjson, vector<HostAddr>& stream_list);
public:
    const string& GetSessionId();
    SessionType GetSessionType();
    void ChangeSessionType(SessionType session_type);
    int GetSessionState();
    SDeviceChannel GetDC();
    CHostInfo GetStreamServer();
private:
    boost::recursive_mutex lock_;
    SDeviceChannel dc_;
    string session_id_;
    SessionType session_type_;

    CHostInfo hi_stream_serv_;
    map<CHostInfo,IMediaSessionSink_ptr> user_ctxs_;

    SessionState session_state_;
    tick_t session_state_tick_;
};

typedef boost::shared_ptr<MediaSession> MediaSession_ptr;

#endif // __MEDIA_SESSION__
