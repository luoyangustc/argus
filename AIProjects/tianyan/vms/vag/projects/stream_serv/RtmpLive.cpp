#include <stdio.h>
#include <string.h>
#include <vector>
#include "tick.h"
#include "RtmpLive.h"
#include "logging_posix.h"
#include "sps_decode.h"

#define RTMP_HEAD_SIZE   (sizeof(RTMPPacket)+RTMP_MAX_HEADER_SIZE)

static int ConvH264ToAvc(unsigned char*  pH264Data, int iH264Size, NaluUnit*  pNalArray, int iNalMaxCount, int*  piNalCount);
static int SerializeAvcToBuf(NaluUnit*  pNalArray, int iNalCount, unsigned char*  pBuf, int* piSize);

CRtmpLive::CRtmpLive()
    : m_nLastRestartTick(0)
    , m_bIsConnected(false)
    , last_audio_timestamp_(-1)
    , last_video_timestamp_(-1)
    , audio_timestamp_offset_(0)
    , video_timestamp_offset_(0)
    , first_audio_flag_(false)
    , first_video_flag_(false)
{
#if DUMP_FILE == 1
    m_pVideoDat = NULL;
    m_pVideoH264 = NULL;
	m_pVideoNalu = NULL;
#endif
}

CRtmpLive::~CRtmpLive()
{
    Stop();
}

int CRtmpLive::SetAudioInfo(MI_AudioInfo audio_info)
{
    if( audio_info.codec_fmt != MI_AUDIO_AAC && 
        audio_info.codec_fmt != MI_AUDIO_G711_A && 
        audio_info.codec_fmt != MI_AUDIO_G711_U && 
        audio_info.codec_fmt != MI_AUDIO_MP3 )
    {
        return -1;
    }

    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    has_audio_ = true;
    audio_info_ = audio_info;

    Debug( "rtmp_url(%s),au_codec(%d),au_sample(%d),au_bitwidth(%d),au_chanenl(%d)", 
        m_strRtmpUrl.c_str(), 
        (int)audio_info_.codec_fmt, (int)audio_info_.sample, (int)audio_info_.bitwidth, (int)audio_info_.channel);
    if( audio_info_.sepc_size > 1 )
    {
        Debug( "rtmp_url(%s),au_sepc_len(%d),au_sepc_data(0x%02x 0x%02x)", 
            m_strRtmpUrl.c_str(), (int)audio_info_.sepc_size,
            (int)audio_info_.sepc_data[0], (int)audio_info_.sepc_data[1] );
    }
    return true;
}

int CRtmpLive::Start(const char* url)
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);

    m_strRtmpUrl = url;

    m_nRtmpBodyBuffSize = 1024*1024;
    m_RtmpBodyBuff.reset(new unsigned char[m_nRtmpBodyBuffSize]);
    if ( !m_RtmpBodyBuff.get() )
    {
        m_nRtmpBodyBuffSize = 0;
        return -2;
    }

    m_pRtmp = RTMP_Alloc();
    if ( !m_pRtmp )
    {
        return -3;
    }
    RTMP_Init(m_pRtmp);

    if ( !RTMP_SetupURL(m_pRtmp,(char*)url) )
    {
        RTMP_Free(m_pRtmp);
		m_pRtmp = NULL;
        return -4;
    }

    RTMP_EnableWrite(m_pRtmp);

    if ( !RTMP_Connect(m_pRtmp, NULL) ) 
    {
        RTMP_Free(m_pRtmp);
		m_pRtmp = NULL;
        return -5;
    } 

    if ( !RTMP_ConnectStream(m_pRtmp,0) )
    {
        RTMP_Close(m_pRtmp);
        RTMP_Free(m_pRtmp);
		m_pRtmp = NULL;
        return -6;
    }

	m_bIsConnected = true;
    //m_nLastVideoTick = 0;
    //m_nVideoTimestamp =0;
	
#if DUMP_FILE == 1
    char* pch = (char*)strrchr(url,'/');
    if( pch )
    {
        string source(pch+1);
        string file_name = source + ".dat";
        m_pVideoDat = fopen(file_name.c_str(), "wb");

        file_name = source + ".264";
        m_pVideoH264 = fopen(file_name.c_str(), "wb");

        file_name = source + ".nalu";
        m_pVideoNalu = fopen(file_name.c_str(), "wb");
    }
#endif

    Debug( "rtmp live agent start, url=%s", m_strRtmpUrl.c_str() );
    return 0;
}

int CRtmpLive::Stop()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if(m_pRtmp)
    {
        RTMP_Close(m_pRtmp);
        RTMP_Free(m_pRtmp);
        m_pRtmp = NULL;
		m_bIsConnected = false;
    }

#if DUMP_FILE == 1
    if ( m_pVideoDat )
    {
        fclose ( m_pVideoDat );
        m_pVideoDat = NULL;
    };

    if ( m_pVideoH264 )
    {
        fclose ( m_pVideoH264 );
        m_pVideoH264 = NULL;
    };

	if (m_pVideoNalu)
	{
		fclose(m_pVideoNalu);
		m_pVideoNalu = NULL;
	}
#endif
    Debug( "rtmp live agent stop, url=%s", m_strRtmpUrl.c_str() );
    return 0;
}

int CRtmpLive::Restart()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
	bool need_conn_flag = false;
	tick_t curr = get_current_tick();
	if(m_nLastRestartTick == 0)
	{
		need_conn_flag = true;
	}	
	else if(curr-m_nLastRestartTick > 5*1000)
	{
		need_conn_flag = true;
	}

	if(!need_conn_flag)
	{
		return 1;
	}

	m_nLastRestartTick = curr;

    if(m_pRtmp)
    {
        RTMP_Close(m_pRtmp);
        RTMP_Free(m_pRtmp);
        m_pRtmp = NULL;
    }

	m_pRtmp = RTMP_Alloc();
	if ( !m_pRtmp )
	{
		return -2;
	}
	RTMP_Init(m_pRtmp);

	if(RTMP_SetupURL(m_pRtmp,(char*)m_strRtmpUrl.c_str()) == FALSE)
	{
		RTMP_Free(m_pRtmp);
		m_pRtmp = NULL;
		return -3;
	}

	RTMP_EnableWrite(m_pRtmp);

	if (RTMP_Connect(m_pRtmp, 0) == FALSE) 
	{
		RTMP_Free(m_pRtmp);
		m_pRtmp = NULL;
		return -4;
	} 

	if (RTMP_ConnectStream(m_pRtmp,0) == FALSE)
	{
		RTMP_Close(m_pRtmp);
		RTMP_Free(m_pRtmp);
		m_pRtmp = NULL;
		return -5;
	}

	m_bIsConnected = true;
	//m_nLastVideoTick = 0;
	//m_nVideoTimestamp =0;
    return 0;
}

bool CRtmpLive::IsReconnectRtmp()
{
	bool bRes = false;
	boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
	if (m_isReconnectFlag!=m_nLastRestartTick)
	{
		m_isReconnectFlag = m_nLastRestartTick;
		bRes = true;
	}
	return bRes;
}

void CRtmpLive::AlignTimestamp(MI_FrameData_ptr frame_data, uint32_t& out_frame_ts)
{
    if (frame_data->is_audio_)
    {
        last_audio_timestamp_ = frame_data->frame_ts_;
    }
    else
    {
        last_video_timestamp_ = frame_data->frame_ts_;
    }

    if ( !frame_data->is_audio_ )
    {   
        if ( !first_video_flag_)
        {
            first_video_flag_ = true;
            if( -1 != last_audio_timestamp_ )
            {            
                video_timestamp_offset_ = frame_data->frame_ts_ - last_audio_timestamp_;
            }
        }
        else if ( frame_data->frame_ts_ < 5000 && (frame_data->frame_ts_ + 5000) < last_audio_timestamp_ )
        {
            video_timestamp_offset_ = 0;
            first_audio_flag_ =false;
        }

        last_audio_timestamp_ = frame_data->frame_ts_;
        out_frame_ts = frame_data->frame_ts_ - video_timestamp_offset_;
    }
    else
    {       
        if (! first_audio_flag_)
        {
            first_audio_flag_ = true;
            if ( -1 != last_video_timestamp_ )
            {
                audio_timestamp_offset_ = (uint32_t)frame_data->frame_ts_ - (uint32_t)last_video_timestamp_;
            }
        }
        else if ( frame_data->frame_ts_ < 5000 && (frame_data->frame_ts_ + 5000) < last_video_timestamp_ )
        {
            audio_timestamp_offset_ = 0;
            first_video_flag_=false;
        }

        last_video_timestamp_ = frame_data->frame_ts_;
        out_frame_ts = frame_data->frame_ts_ - audio_timestamp_offset_;
    }
}

int CRtmpLive::OnStream(MI_FrameData_ptr frame_data)
{
    int ret = -1;

    {
        boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
        if(frame_data->frame_size_ > m_nRtmpBodyBuffSize)
        {
            uint32_t new_buffer_size = frame_data->frame_size_ + 512;
            boost::shared_array<unsigned char> new_body_buff(new unsigned char[new_buffer_size]);
            if(!new_body_buff)
            {
                return -1;
            }
            m_RtmpBodyBuff = new_body_buff;
            m_nRtmpBodyBuffSize = new_buffer_size;
        }
    }

    uint32_t frame_ts = frame_data->frame_ts_;
    AlignTimestamp(frame_data, frame_ts);

    unsigned char* buff = frame_data->data_.get();
    int buff_size = frame_data->frame_size_;
    if( frame_data->is_audio_ ) {
        ret = this->OnAudio( buff, buff_size, frame_ts );
    }
    else {
        NaluUnit nals[64];
        int nal_idx = 0;
#if 1
        ConvH264ToAvc(buff, buff_size, nals, 64, &nal_idx);

        for (int i = 0; i < nal_idx; i++)
        {
            if (nals[i].type == EN_NAL_SPS)
            {
                ret = this->OnSps(nals[i].data, nals[i].size);
            }

            if (nals[i].type == EN_NAL_PPS)
            {
                ret = this->OnPps(nals[i].data, nals[i].size);
            }

        }

        //发送视频
        {
            bool isKeyFrm = (frame_data->frame_type_ == MI_FRAME_I)?true:false;
            ret = this->OnVideo2(nals, nal_idx, frame_ts, isKeyFrm);
        }
#else
        while(buff_size > 0)
        {
            ret = this->ReadOneNalu(buff, buff_size, nals[nal_idx]);
            if( ret < 0 ){
                Error( "read nal failed, ret = %d!", ret );
                break;
            }

            Trace( "nal_type(%d), nal_size(%d), nal_data(%p), frm_ts(%u), ret=%d!", 
                nals[nal_idx].type, nals[nal_idx].size, nals[nal_idx].data, frame_ts, ret );

            buff += ret;
            buff_size -= ret;

            if(nals[nal_idx].type==EN_NAL_SPS){
                ret = this->OnSps(nals[nal_idx].data, nals[nal_idx].size);
            }
            else if(nals[nal_idx].type==EN_NAL_PPS){
                ret = this->OnPps(nals[nal_idx].data, nals[nal_idx].size);
            }

            nal_idx++;
            if( nal_idx >= sizeof(nals)/sizeof(NaluUnit) ) 
            {
                Error( "nal num too more!");
                ret = -2;
                break;
            }
        }
        
        //发送视频
        {
            bool isKeyFrm = (frame_data->frame_type_ == MI_FRAME_I)?true:false;
            ret = this->OnVideo2(nals, nal_idx, frame_ts, isKeyFrm);
        }
#endif
    }

    return ret;
}

int CRtmpLive::OnSps(const void* sps, uint32_t sps_len)
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( !m_RtmpMetaData.Sps.get() )
    {
        m_RtmpMetaData.Sps.reset(new unsigned char[sps_len]);
    }
    else if(m_RtmpMetaData.nSpsLen < sps_len)
    {
        m_RtmpMetaData.Sps.reset(new unsigned char[sps_len]);
    }
    m_RtmpMetaData.nSpsLen = sps_len;
    memcpy(m_RtmpMetaData.Sps.get(), sps, sps_len);

    {
        // 解码SPS,获取视频图像宽、高信息   
        int width = 0,height = 0, fps=0;  
        h264_decode_sps((BYTE*)sps, sps_len, width, height, fps); 
        if(fps)
        {
            m_RtmpMetaData.nFrameRate = fps;
        }
        else
        {
            m_RtmpMetaData.nFrameRate = 25;
        }
        m_RtmpMetaData.nWidth = width;
        m_RtmpMetaData.nHeight = height;
    }
    


#if DUMP_FILE == 1
    fwrite(m_RtmpMetaData.Sps.get(), 1, sps_len, m_pVideoH264);
#endif
    return 0;
}

int CRtmpLive::OnPps(const void* pps, uint32_t pps_len)
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( !m_RtmpMetaData.Pps.get() )
    {
        m_RtmpMetaData.Pps.reset(new unsigned char[pps_len]);
    }
    else if(m_RtmpMetaData.nPpsLen < pps_len)
    {
        m_RtmpMetaData.Pps.reset(new unsigned char[pps_len]);
    }
    m_RtmpMetaData.nPpsLen = pps_len;
    memcpy(m_RtmpMetaData.Pps.get(), pps, pps_len);

#if DUMP_FILE == 1
    fwrite(m_RtmpMetaData.Pps.get(), 1, pps_len, m_pVideoH264);
#endif

    return 0;
}

int CRtmpLive::OnVideo(const void* data, uint32_t data_len, uint32_t timestamp, bool is_key_frame)
{
    if(data == NULL && data_len<11)
    {  
        return -1;
    }
    
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);

	int nRet = -1;
	if( m_bIsConnected)
	{
		if(!RTMP_IsConnected(m_pRtmp))
		{
			m_bIsConnected = false;
		}
	}

	if(!m_bIsConnected)
	{
        Warn( "rtmp live agent restart, url=%s", m_strRtmpUrl.c_str() );
		return Restart();
	}

    if(is_key_frame)
    {
        SendSpsAndPps();
    }

    unsigned char *body = m_RtmpBodyBuff.get();
    unsigned int body_size = 0;
    int i = 0; 
    if(is_key_frame)
    {  
        *body++ = 0x17;// 1:Iframe  7:AVC
    }
    else
    {  
        *body++ = 0x27;// 2:Pframe  7:AVC
    }
    *body++ = 0x01;// AVC NALU   
    *body++ = 0x00;
    *body++ = 0x00;
    *body++ = 0x00;
    // NALU size
    *body++ = (data_len>>24) & 0xff;
    *body++ = (data_len>>16) & 0xff;
    *body++ = (data_len>>8) & 0xff;
    *body++ = (data_len) & 0xff;
    // NALU data   
    memcpy(body, data, data_len);
    body += data_len;
    body_size = body - m_RtmpBodyBuff.get();

    /*if( (m_nLastVideoTick == 0) || (timestamp < m_nLastVideoTick) )
    {
        m_nVideoTimestamp = 0;
        m_nLastVideoTick = timestamp;//RTMP_GetTime();
    }
    else
    {
		m_nVideoTimestamp += timestamp-m_nLastVideoTick;
		m_nLastVideoTick = timestamp;
    }*/

    nRet = SendPacket(RTMP_PACKET_TYPE_VIDEO, m_RtmpBodyBuff.get(), body_size, timestamp);
    return nRet;
}

int CRtmpLive::OnVideo2(NaluUnit* nals, uint32_t nals_num, uint32_t timestamp, bool is_key_frame)
{
    if( nals == NULL && nals_num==0 )
    {  
        return -1;
    }

    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( m_bIsConnected)
    {
        if(!RTMP_IsConnected(m_pRtmp))
        {
            m_bIsConnected = false;
        }
    }

    if(!m_bIsConnected)
    {
        Warn( "rtmp live agent restart, url=%s", m_strRtmpUrl.c_str() );
        (void)Restart();
        return -2;
    }

    if(is_key_frame)
    {
        SendSpsAndPps();
    }

    unsigned char *body = m_RtmpBodyBuff.get();
    unsigned int body_size = 0;
    int i = 0; 
    if(is_key_frame)
    {  
        *body++ = 0x17;// 1:Iframe  7:AVC
    }
    else
    {  
        *body++ = 0x27;// 2:Pframe  7:AVC
    }
    *body++ = 0x01;// AVC NALU   
    *body++ = 0x00;
    *body++ = 0x00;
    *body++ = 0x00;

    for( int i=0; i<nals_num; i++ )
    {
        uint8_t* data = nals[i].data;
        uint32_t data_len = nals[i].size;

        // NALU size
        *body++ = (data_len>>24) & 0xff;
        *body++ = (data_len>>16) & 0xff;
        *body++ = (data_len>>8) & 0xff;
        *body++ = (data_len) & 0xff;
        // NALU data   
        memcpy(body, data, data_len);
        body += data_len;
    }

    body_size = body - m_RtmpBodyBuff.get();

    /*if( (m_nLastVideoTick == 0) || (timestamp < m_nLastVideoTick) )
    {
        m_nVideoTimestamp = 0;
        m_nLastVideoTick = timestamp;//RTMP_GetTime();
    }
    else
    {
		m_nVideoTimestamp += timestamp-m_nLastVideoTick;
		m_nLastVideoTick = timestamp;
    }*/

    return SendPacket(RTMP_PACKET_TYPE_VIDEO, m_RtmpBodyBuff.get(), body_size, timestamp);
}

int CRtmpLive::OnAudio(unsigned char* raw_data, uint32_t data_len, uint32_t timestamp)
{
	if( !raw_data || !data_len )
	{  
		return -1;
	}
    
	boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    if( !has_audio_ )
    {
        return -2;
    }

	if( m_bIsConnected )
	{
		if(!RTMP_IsConnected(m_pRtmp))
		{
			m_bIsConnected = false;
		}
	}

	if(!m_bIsConnected)
	{
		return -3;
	}

    if( audio_info_.codec_fmt == MI_AUDIO_AAC )
    {
        SendAACSpecificInfo();
    }

#if DUMP_FILE == 1
	fwrite(raw_data, 1, data_len, m_pVideoNalu);
	int iLength = data_len;
#endif

	unsigned char *body = m_RtmpBodyBuff.get();
    uint8_t flv_au_fmt =GetFlvAudioFmt(audio_info_.codec_fmt);
    uint8_t flv_au_sample = GetFlvSampleRateIndex(audio_info_.sample);

    //音频格式 4bits | 采样率 2bits | 采样精度 1bits | 声道数 1bits|
    *body++ = (flv_au_fmt<<4) | (flv_au_sample<<2) | (audio_info_.bitwidth<<1) |  audio_info_.channel;

    //AAC packet type
    *body++ = 0x01; //0x00:AAC sequence header, 0x01:AAC raw data

	memcpy(body, raw_data, data_len);
	body += data_len;

	unsigned int body_size = body - m_RtmpBodyBuff.get();
	int nRet = SendPacket(RTMP_PACKET_TYPE_AUDIO, m_RtmpBodyBuff.get(), body_size, timestamp);
	return nRet;
}

int CRtmpLive::SendSpsAndPps()
{
    int nRet = 0;
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
    unsigned int sps_len = m_RtmpMetaData.nSpsLen;
    unsigned int pps_len = m_RtmpMetaData.nPpsLen;
    if( !sps_len || !pps_len )
    {
        return -1;
    }

    unsigned char* sps = m_RtmpMetaData.Sps.get();
    unsigned char* pps = m_RtmpMetaData.Pps.get();
    if( !sps || !pps )
    {
        return -2;
    }

    unsigned char* body = m_RtmpBodyBuff.get();
    unsigned int body_size = 0;

    *body++ = 0x27; // 2:inter-frame  7:AVC  
    *body++ = 0x00; // AVC sequence header

    *body++ = 0x00;
    *body++ = 0x00;
    *body++ = 0x00;

    /*AVCDecoderConfigurationRecord*/
    *body++ = 0x01;
    *body++ = sps[1];
    *body++ = sps[2];
    *body++ = sps[3];
    *body++ = 0xff;

    /*sps*/
    *body++ = 0xe1;
    *body++ = (sps_len >> 8) & 0xff;
    *body++ = sps_len & 0xff;
    memcpy(body, sps, sps_len);
    body += sps_len;

    /*pps*/
    *body++ = 0x01;
    *body++ = (pps_len >> 8) & 0xff;
    *body++ = pps_len & 0xff;
    memcpy(body, pps, pps_len);
    body += pps_len;
    body_size = body - m_RtmpBodyBuff.get();
    nRet = SendPacket(RTMP_PACKET_TYPE_VIDEO, m_RtmpBodyBuff.get(), body_size, 0);

    return nRet;
}
 
int CRtmpLive::SendAACSpecificInfo()
{
    boost::lock_guard<boost::recursive_mutex> lock(m_Lock);
	if( !has_audio_ ||
        audio_info_.codec_fmt != MI_AUDIO_AAC || 
        !audio_info_.sepc_size )
    {
		return -1;
    }

	if( audio_info_.sepc_size > m_nRtmpBodyBuffSize )
	{
		uint32_t new_buffer_size = audio_info_.sepc_size + 512;
		boost::shared_array<unsigned char> new_body_buff(new unsigned char[new_buffer_size]);
		m_RtmpBodyBuff = new_body_buff;
		m_nRtmpBodyBuffSize = new_buffer_size;
	}

	unsigned char *body = m_RtmpBodyBuff.get();
    uint8_t flv_au_fmt =GetFlvAudioFmt(audio_info_.codec_fmt);
    uint8_t flv_au_sample = GetFlvSampleRateIndex(audio_info_.sample);
    
    //音频格式 4bits | 采样率 2bits | 采样精度 1bits | 声道数 1bits|
    *body++ = (flv_au_fmt<<4) | (flv_au_sample<<2) | (audio_info_.bitwidth<<1) |  audio_info_.channel;

    //AAC packet type
    *body++ = 0x00; //0x00:AAC sequence header, 0x01:AAC raw data

    //AAC specific data
	memcpy(body, audio_info_.sepc_data.data(), audio_info_.sepc_size);
	body += audio_info_.sepc_size;
	
    {
        unsigned char* p = m_RtmpBodyBuff.get();
        Debug("rtmp_url(%s), audio-->0x%02x 0x%02x 0x%02x 0x%02x",
            m_strRtmpUrl.c_str(), (int)p[0], (int)p[1], (int)p[2], (int)p[3]);
    }
    

	int nRet = SendPacket(RTMP_PACKET_TYPE_AUDIO, m_RtmpBodyBuff.get(), body-m_RtmpBodyBuff.get(), 0);
	return nRet;
}

int CRtmpLive::SendPacket(int packet_type, const void* data, uint32_t data_size, uint32_t timestamp)
{
    RTMPPacket packet;
    RTMPPacket_Reset(&packet);
    RTMPPacket_Alloc(&packet, data_size);

    packet.m_nBodySize = data_size;
    memcpy(packet.m_body,data,data_size);

    packet.m_hasAbsTimestamp = 0;
    packet.m_packetType = packet_type; /*此处为类型有两种一种是音频,一种是视频*/
	packet.m_nInfoField2 = m_pRtmp->m_stream_id;
	packet.m_nChannel = 0x04;
	if (RTMP_PACKET_TYPE_AUDIO == packet_type)
	{
		packet.m_nChannel = 0x05;
	}

    packet.m_headerType = RTMP_PACKET_SIZE_LARGE;
    if( RTMP_PACKET_TYPE_AUDIO == packet_type && data_size !=4 )
    {
        packet.m_headerType = RTMP_PACKET_SIZE_MEDIUM;
    }
    packet.m_nTimeStamp = timestamp;

	int nRet = RTMP_SendPacket(m_pRtmp, &packet, TRUE);
    RTMPPacket_Free(&packet);

#if DUMP_FILE == 1
	if (packet_type == RTMP_PACKET_TYPE_AUDIO)
		return nRet;
	
    char videinfo[1024];
    memset(videinfo, 0x0, sizeof(videinfo));

    unsigned char* body_pos = m_RtmpBodyBuff.get();
    std::string frm_info;
    if(body_pos[0]==0x17)
    {
        frm_info = "I Frame";
    }
    else if (body_pos[0]==0x27)
    {
        frm_info = "P Frame";
    }
    else
    {
        frm_info = "Unknow";
    }

    char* pos = videinfo;
    pos += sprintf( pos, "Time:%u, body_size:%d, %s-->\n", packet.m_nTimeStamp, packet.m_nBodySize, frm_info.c_str());

    for( int i =0; i<32; ++i)
    {
        if( (i % 16) == 0 )
        {
            *pos++ = '\n';
        }
        pos += sprintf(pos, "%02x ", *(m_RtmpBodyBuff.get()+i));
    }
    *pos++ = '\n';
    fwrite(videinfo, 1, strlen(videinfo), m_pVideoDat);
#endif

    return nRet;
}

int CRtmpLive::ReadOneNalu(const void* data, uint32_t data_len, NaluUnit& nalu)
{
    int read_size = 0;

    if(data_len < 4)
    {
        return -1;
    }

    unsigned char* s = (unsigned char*)data;
    unsigned char* p = (unsigned char*)data;
    if(p[0]==0x00 && p[1]==0x00 && p[2]==0x01)
    {
        p += 3;
    }
    else if(p[0]==0x00 &&p[1]==0x00 && p[2]==0x00 && p[3]==0x01)
    {
        p += 4;
    }
    else
    {
        return -2;
    }

    nalu.type = (*p)&0x1f;
    nalu.data = p;

    bool bFound = false;
    while( p <= s+data_len-4 ) //预留结尾的4个byte，防止检查越界
    {
        if(p[0]==0x00 && p[1]==0x00 && p[2]==0x01)
        {
            bFound = true;
            break;
        }
        else if(p[0]==0x00 &&p[1]==0x00 && p[2]==0x00 && p[3]==0x01)
        {
            bFound = true;
            break;
        }
        else
        {
            p++;
        }
    }

    if( bFound )
    {
        nalu.size = p - nalu.data;
        read_size = p - s;
    }
    else
    {
        nalu.size = s + data_len - nalu.data;
        read_size = s + data_len - s;
    }

    return read_size;
}

uint8_t CRtmpLive::GetFlvSampleRateIndex(uint8_t sample_rate)
{
	uint8_t index=0;
	switch (sample_rate)
	{
	case MI_AUDIO_SR_8_KHZ:
		index=0;
		break;
	case MI_AUDIO_SR_11_025_KHZ:
    case MI_AUDIO_SR_12_KHZ:
    case MI_AUDIO_SR_16_KHZ:
		index=1;
		break;
	case MI_AUDIO_SR_22_05_KHZ:
    case MI_AUDIO_SR_24_KHZ:
    case MI_AUDIO_SR_32_KHZ:
		index=2;
		break;
	case MI_AUDIO_SR_44_1_KHZ:
    case MI_AUDIO_SR_48_KHZ:
    case MI_AUDIO_SR_64_KHZ:
    case MI_AUDIO_SR_88_2_KHZ:
    case MI_AUDIO_SR_96_KHZ:
		index=3;
		break;
	default:
		break;
	}
	return index;
}

uint8_t CRtmpLive::GetFlvAudioFmt(uint8_t codec_type)
{
    uint8_t audio_fmt = 0;
    switch (codec_type)
    {
    case MI_AUDIO_AAC:
        audio_fmt = 0x0A;
        break;
    case MI_AUDIO_G711_A:
        audio_fmt = 0x07;
        break;
    case MI_AUDIO_G711_U:
        audio_fmt = 0x08;
        break;
    case MI_AUDIO_MP3:
        audio_fmt = 0x02;
        break;
    }
    return audio_fmt;
}


static int ConvH264ToAvc(unsigned char*  pH264Data, int iH264Size, NaluUnit*  pNalArray, int iNalMaxCount, int*  piNalCount)
{
    int iRet = 0;
    unsigned char*  pEnd = NULL;
    unsigned char*  pFind = NULL;
    unsigned  char aSyncL[4] = { 0, 0, 0, 1 };
    unsigned  char aSyncS[3] = { 0, 0, 1 };
    int iNalCurCount = 0;

    do 
    {
        if (pH264Data == NULL || iH264Size <= 4 || pNalArray == NULL)
        {
            iRet = 1;
        }

        pEnd = pH264Data + iH264Size - 4;
        pFind = pH264Data;
        while (pFind < pEnd)
        {
            if (memcmp(pFind, aSyncL, 4) == 0)
            {
                if (iNalCurCount != 0)
                {
                    pNalArray[iNalCurCount - 1].size = pFind - pNalArray[iNalCurCount - 1].data;
                }

                pFind += 4;
                if (iNalCurCount < iNalMaxCount)
                {
                    pNalArray[iNalCurCount].data = pFind;
                    pNalArray[iNalCurCount].type = (*pFind) & 0x1f;
                }

                iNalCurCount++;
            }
            else if (memcmp(pFind, aSyncS, 3) == 0)
            {
                if (iNalCurCount != 0)
                {
                    pNalArray[iNalCurCount - 1].size = pFind - pNalArray[iNalCurCount - 1].data;
                }

                pFind += 3;
                if (iNalCurCount < iNalMaxCount)
                {
                    pNalArray[iNalCurCount].data = pFind;
                    pNalArray[iNalCurCount].type = (*pFind) & 0x1f;
                }

                iNalCurCount++;
            }
            else
            {
                pFind++;
            }
        }

        if (iNalCurCount > 0)
        {
            pNalArray[iNalCurCount - 1].size = pH264Data + iH264Size - pNalArray[iNalCurCount - 1].data;
        }

        *piNalCount = iNalCurCount;
    } while (0);

    return 0;
}

static int SerializeAvcToBuf(NaluUnit*  pNalArray, int iNalCount, unsigned char*  pBuf, int* piSize)
{
    int iSize = 0;
    int iIndex = 0;
    unsigned char*   pCurbuf;

    pCurbuf = pBuf;
    for (iIndex = 0; iIndex < iNalCount;iIndex++)
    {
        // NALU size
        *pCurbuf++ = (pNalArray[iIndex].size >> 24) & 0xff;
        *pCurbuf++ = (pNalArray[iIndex].size >> 16) & 0xff;
        *pCurbuf++ = (pNalArray[iIndex].size >> 8) & 0xff;
        *pCurbuf++ = (pNalArray[iIndex].size) & 0xff;

        iSize += 4;
        memcpy(pCurbuf, pNalArray[iIndex].data, pNalArray[iIndex].size);
        pCurbuf += pNalArray[iIndex].size;
        iSize += pNalArray[iIndex].size;
    }

    *piSize = iSize;
    return 0;
}
