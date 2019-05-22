#ifndef __MEDIA_INFO_H__
#define __MEDIA_INFO_H__

//#include "typedefine.h"
#include <stdint.h>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

// ֡����
enum MI_FrameType
{
    MI_FRAME_I       = 0x01,    //I֡
    MI_FRAME_P       = 0x02,    //P֡
    MI_FRAME_AUDIO   = 0x03,    //��Ƶ֡
};

enum MI_VideoCodecType 
{
    MI_VIDEO_H264 = 0,
    MI_VIDEO_H265
};

enum MI_AudioCodecType 
{
    MI_AUDIO_AAC = 0,
    MI_AUDIO_G711_A,
    MI_AUDIO_G711_U,
    MI_AUDIO_MP3
};

enum MI_AudioChannelType
{
    MI_AUDIO_CH_MONO           = 0,    // ������
    MI_AUDIO_CH_STEREO         = 1,    // ��������
};

// ��Ƶλ�����Ͷ���
enum MI_AudioBitwidthType
{
    MI_AUDIO_BW_8BIT           = 0,    //λ��8bit
    MI_AUDIO_BW_16BIT          = 1,    //λ��16bit
};

// ��Ƶ���������Ͷ���
enum MI_AudioSampleRateType
{
    MI_AUDIO_SR_8_KHZ       = 0,    // �����ʣ�8khz
    MI_AUDIO_SR_11_025_KHZ  = 1,    // �����ʣ�11.025khz
    MI_AUDIO_SR_12_KHZ      = 2,    // �����ʣ�12khz
    MI_AUDIO_SR_16_KHZ      = 3,    // �����ʣ�16khz
    MI_AUDIO_SR_22_05_KHZ   = 4,    // �����ʣ�22.05khz
    MI_AUDIO_SR_24_KHZ      = 5,    // �����ʣ�24khz
    MI_AUDIO_SR_32_KHZ      = 6,    // �����ʣ�32khz
    MI_AUDIO_SR_44_1_KHZ    = 7,    // �����ʣ�44.1khz
    MI_AUDIO_SR_48_KHZ      = 8,    // �����ʣ�48khz
    MI_AUDIO_SR_64_KHZ      = 9,    // �����ʣ�64khz
    MI_AUDIO_SR_88_2_KHZ    = 10,   // �����ʣ�88.2khz
    MI_AUDIO_SR_96_KHZ      = 11    // �����ʣ�96khz
};

struct MI_FrameData
{
    bool is_audio_;
    bool is_i_frame_;
    bool is_frist_;
    uint8_t frame_type_;
    uint32_t frame_seq_;
    uint32_t frame_ts_;
    uint32_t frame_base_time_;    //sec
    uint32_t frame_size_;
    uint32_t frame_av_seq_;
    uint32_t crc32_hash_;
    boost::shared_array<uint8_t> data_;

    MI_FrameData()
    {
        is_audio_ = false;
        is_i_frame_ = false;
        is_frist_ = false;
        frame_type_ = 0x01;
        frame_seq_ = 0;
        frame_ts_ = 0;
        frame_base_time_ = 0;
        frame_size_ = 0;
        frame_av_seq_ = 0;
        crc32_hash_ = 0;
    }
};

typedef boost::shared_ptr<MI_FrameData> MI_FrameData_ptr;

struct MI_VideoInfo 
{
    uint8_t encode_type_; // refer to "MI_VideoCodecType"
};

struct MI_AudioInfo
{
    uint8_t codec_fmt;            // ��Ƶ��ʽ, �ο� "MI_AudioCodecType"
    uint8_t channel;              // ��Ƶͨ��, �ο� "MI_AudioChannelType"
    uint8_t sample;               // ��Ƶ������, �ο� "MI_AudioSampleRateType"
    uint8_t bitwidth;             // λ��, �ο� "MI_AudioBitwidthType"
    uint8_t sepc_size;            // ��Ƶ��ϸ��Ϣ����
    std::vector<uint8_t> sepc_data;    // ��Ƶ��ϸ��Ϣ,����ο��ĵ�
};

struct MI_MediaDesc
{
    MI_VideoInfo video;
    MI_AudioInfo audio;
};

#endif //__MEDIA_INFO_H__
