#ifndef __MEDIA_SESSION_MGR_
#define __MEDIA_SESSION_MGR_

#include <string>
#include <map>
#include "MediaSession.h"
#include "protocol/include/protocol_device.h"

using namespace std;
using namespace protocol;

class CMediaSessionMgr
{
public:
	CMediaSessionMgr() {}
	~CMediaSessionMgr() {}
    void Update();

    MediaSession_ptr GetMediaSession( const SDeviceChannel& dc, SessionType session_type, bool is_create = false );
    MediaSession_ptr GetMediaSession( const string& session_id);
    MediaSession_ptr CreateMediaSession( const SDeviceChannel& dc, SessionType session_type );
    MediaSession_ptr RebuildMediaSession(const SDeviceChannel& dc, SessionType session_type, string session_id, HostAddr stream_addr);
    void RemoveMediaSession( const string& session_id);
    void RemoveMediaSession(const SDeviceChannel& dc, SessionType session_type);
public:
    bool OnUserMediaOpenReq(const SDeviceChannel& dc, SessionType session_type, const CHostInfo& hiRemote, IMediaSessionSink_ptr user_ctx);
    bool OnUserClose(const SDeviceChannel& dc, SessionType session_type, const CHostInfo& hiRemote);
    bool OnUserClose(string session_id, const CHostInfo& hiRemote);
    bool OnDeviceClose( const string& device_id );
public:
    bool OnDeviceStatusReport(const vector<DeviceMediaSessionStatus>& status_list);
    bool OnDeviceMediaOpenAck(const DeviceMediaOpenResp& resp);
private:
    boost::recursive_mutex lock_;
    map<SDeviceChannel, MediaSession_ptr> live_sessions_;
    map<SDeviceChannel, MediaSession_ptr> playback_sessions_;
    map<string, MediaSession_ptr> sessions_;    // session_id to media_session
};

typedef boost::shared_ptr<CMediaSessionMgr> CMediaSessionMgr_ptr;

#endif // __MEDIA_SESSION_MGR_
