#ifndef __MEDIA_INFO_H__
#define __MEDIA_INFO_H__

//#include "typedefine.h"
#include <stdint.h>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

// 帧类型
enum MI_FrameType
{
    MI_FRAME_I       = 0x01,    //I帧
    MI_FRAME_P       = 0x02,    //P帧
    MI_FRAME_AUDIO   = 0x03,    //音频帧
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
    MI_AUDIO_CH_MONO           = 0,    // 单声道
    MI_AUDIO_CH_STEREO         = 1,    // 立体声道
};

// 音频位宽类型定义
enum MI_AudioBitwidthType
{
    MI_AUDIO_BW_8BIT           = 0,    //位宽：8bit
    MI_AUDIO_BW_16BIT          = 1,    //位宽：16bit
};

// 音频采样率类型定义
enum MI_AudioSampleRateType
{
    MI_AUDIO_SR_8_KHZ       = 0,    // 采样率：8khz
    MI_AUDIO_SR_11_025_KHZ  = 1,    // 采样率：11.025khz
    MI_AUDIO_SR_12_KHZ      = 2,    // 采样率：12khz
    MI_AUDIO_SR_16_KHZ      = 3,    // 采样率：16khz
    MI_AUDIO_SR_22_05_KHZ   = 4,    // 采样率：22.05khz
    MI_AUDIO_SR_24_KHZ      = 5,    // 采样率：24khz
    MI_AUDIO_SR_32_KHZ      = 6,    // 采样率：32khz
    MI_AUDIO_SR_44_1_KHZ    = 7,    // 采样率：44.1khz
    MI_AUDIO_SR_48_KHZ      = 8,    // 采样率：48khz
    MI_AUDIO_SR_64_KHZ      = 9,    // 采样率：64khz
    MI_AUDIO_SR_88_2_KHZ    = 10,   // 采样率：88.2khz
    MI_AUDIO_SR_96_KHZ      = 11    // 采样率：96khz
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
    uint8_t codec_fmt;            // 音频格式, 参考 "MI_AudioCodecType"
    uint8_t channel;              // 音频通道, 参考 "MI_AudioChannelType"
    uint8_t sample;               // 音频采样率, 参考 "MI_AudioSampleRateType"
    uint8_t bitwidth;             // 位宽, 参考 "MI_AudioBitwidthType"
    uint8_t sepc_size;            // 音频详细信息长度
    std::vector<uint8_t> sepc_data;    // 音频详细信息,具体参考文档
};

struct MI_MediaDesc
{
    MI_VideoInfo video;
    MI_AudioInfo audio;
};

#endif //__MEDIA_INFO_H__
