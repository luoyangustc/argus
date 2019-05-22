#ifndef __FRAME_H__
#define __FRAME_H__

#include "media_info.h"
#include "CommonInc.h"

enum {
    en_frm_sts_ok               =  0,
    en_frm_sts_mallc_fail       = -1,
    en_frm_sts_crc_err          = -2,
    en_frm_sts_overflow         = -3,
    en_frm_sts_offset_err       = -4,
    en_frm_sts_timeout          = -5,
    en_frm_sts_force_discard    = -6
};

class CFrame
{
public:
    CFrame(const StreamMediaFrameNotify& notify);
    ~CFrame();

    int GetStatus();
    bool IsIFrame(){ return IsKeyFrame(frame_type_);}
    bool IsAudio(){ return IsAudioFrame(frame_type_);}
    uint32 life_time(){return (uint32)(get_current_tick()-add_tick_);}

    void Discard();
    bool IsFull();
    int SaveData(uint32 offset, PBYTE buffer, uint32 data_size);
    bool GetData(MI_FrameData_ptr& pFrameData);
    uint32 GetFrameSize(){return frame_size_;}

private:
    boost::recursive_mutex lock_;
    tick_t add_tick_;
    uint32 recv_size_;
    int status_;
private:
    uint8 frame_type_;	    // 0x01:I֡ 0x02:P֡ 0x03:��Ƶ֡
    uint32 frame_av_seq_;    // ����Ƶ֡���(��Ƶ+��Ƶ�ܺ�)
    uint32 frame_seq_;       // ֡���(��Ƶ����Ƶ֡���)
    uint32 frame_base_time_; // ֡��׼ʱ��(��λ��)
    uint32 frame_ts_;        // ֡ʱ���(��λms)
    uint32 frame_size_;      // ֡��С

    uint32 crc32_hash_;

    CLBitField	m_bf_;
    boost::shared_array<uint8> data_;
};

typedef boost::shared_ptr<CFrame> CFrame_ptr;

#endif