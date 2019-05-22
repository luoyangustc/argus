#ifndef __RTMP_LIVE__
#define __RTMP_LIVE__

#include <stdint.h>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread.hpp>
#include <boost/shared_array.hpp>
#include <string>

typedef int SOCKET;

#include "rtmp/rtmp.h"   
#include "rtmp/rtmp_sys.h"
#include "rtmp/amf.h"
#include "media_info.h"

using namespace std;
using namespace boost;

#define  DUMP_FILE 0

enum EnNaluType
{
	EN_NAL_UNKNOWN     = 0,
	EN_NAL_SLICE       = 1,
	EN_NAL_SLICE_DPA   = 2,
	EN_NAL_SLICE_DPB   = 3,
	EN_NAL_SLICE_DPC   = 4,
	EN_NAL_SLICE_IDR   = 5,    /* ref_idc != 0 */
	EN_NAL_SEI         = 6,    /* ref_idc == 0 */
	EN_NAL_SPS         = 7,
	EN_NAL_PPS         = 8,
	EN_NAL_AUD         = 9,
	EN_NAL_FILLER      = 12,
	/* ref_idc == 0 for 6,9,10,11,12 */
};

struct NaluUnit  
{  
	int type;  
	int size;
	unsigned char *data;  
};

struct RTMPMetadata
{  
    // video, must be h264 type   
    unsigned int nWidth;
    unsigned int nHeight;
    unsigned int nFrameRate;

    unsigned int nSpsLen;
    shared_array<unsigned char> Sps;

    unsigned int nPpsLen;
    shared_array<unsigned char> Pps;
};

class CRtmpLive
{
public:
	CRtmpLive();
	virtual ~CRtmpLive();
    int SetAudioInfo(MI_AudioInfo audio_info);
	int Start(const char* url);
	int Stop();
    int OnStream(MI_FrameData_ptr frame_data);
    string GetRtmpUrl() {return m_strRtmpUrl;}
private:
    int OnSps(const void* sps, uint32_t sps_len);
    int OnPps(const void* pps, uint32_t pps_len);
    int OnVideo(const void* data, uint32_t data_len, uint32_t timestamp, bool is_key_frame);
    int OnVideo2(NaluUnit* nals, uint32_t nals_num, uint32_t timestamp, bool is_key_frame);
	int OnAudio(unsigned char* raw_data, uint32_t data_len, uint32_t timestamp);
	static int ReadOneNalu(const void* data, uint32_t data_len, NaluUnit &nalu);
	bool IsReconnectRtmp();
private:
    int Restart();
    void AlignTimestamp(MI_FrameData_ptr frame_data, uint32_t& out_frame_ts);
    int SendSpsAndPps();
    int SendAACSpecificInfo();
	int SendPacket(int packet_type, const void* data, uint32_t data_size, uint32_t timestamp);
private:
	uint8_t GetFlvSampleRateIndex(uint8_t sample_rate);
    uint8_t GetFlvAudioFmt(uint8_t codec_type);
private:
    boost::recursive_mutex m_Lock;
    RTMP* m_pRtmp;
    string m_strRtmpUrl;
    uint32_t m_nLastVideoTick;
    uint32_t m_nVideoTimestamp;
    RTMPMetadata m_RtmpMetaData;

    uint32_t m_nRtmpBodyBuffSize;
    boost::shared_array<unsigned char> m_RtmpBodyBuff;

	tick_t m_nLastRestartTick;
	bool m_bIsConnected;
	tick_t m_isReconnectFlag;

    //audio info
    bool has_audio_;
    MI_AudioInfo audio_info_;

    //timestamp correct
    bool first_video_flag_;
    bool first_audio_flag_;
    uint32_t last_audio_timestamp_;
    uint32_t last_video_timestamp_;
    uint32_t audio_timestamp_offset_;
    uint32_t video_timestamp_offset_;

#if DUMP_FILE == 1
    FILE	*m_pVideoDat;
    FILE	*m_pVideoH264;
	FILE	*m_pVideoNalu;
#endif
};

typedef boost::shared_ptr<CRtmpLive>    CRtmpLive_ptr;

#endif //__RTMP_LIVE__
