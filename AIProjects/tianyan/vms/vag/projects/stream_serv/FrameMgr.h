#ifndef __FRAME_MGR_H__
#define __FRAME_MGR_H__
#include <sstream>
#include "CommonInc.h"
#include "Frame.h"
#include "variant.h"

using namespace std;

/*
typedef struct tagFrameKey 
{
    uint32 frm_seq;
    uint8 frm_type;

public:
    tagFrameKey(){memset(this, 0, sizeof(tagFrameKey));}
    tagFrameKey(uint32 seq, uint8 type)
    {
        frm_seq = seq;
        frm_type = type;
    }

    const tagFrameKey& operator=(const tagFrameKey& right)
    {
        frm_seq = right.frm_seq;
        frm_type = right.frm_type;
        return *this;
    }

    bool operator==(const tagFrameKey& right)const
    {
        if (frm_seq == right.frm_seq
            && frm_type == right.frm_type)
        {
            return true;
        }
        return false;
    }

    bool operator!=(const tagFrameKey& right)const
    {
        if (frm_seq != right.frm_seq
            || frm_type != right.frm_type)
        {
            return true;
        }
        return false;
    }

    bool operator < (const tagFrameKey& right)const
    {
        do 
        {
            if (frm_seq < right.frm_seq)
            {
                return true;
            }
            else if (frm_seq > right.frm_seq)
            {
                break;
            }

            if (frm_type < right.frm_type)
            {
                return true;
            }
            else if (frm_type > right.frm_type)
            {
                break;
            }

        } while (false);
        return false;
    }

}TFrameKey, *PTFrameKey;
*/

class CFrameMgr
{
public:
    CFrameMgr(SDeviceChannel dc);
    ~CFrameMgr();

    bool IsActive();
    bool IsAudioAlive();

    void Update();
    void DumpInfo(Variant& info);

    bool GetFrameData(uint32 frm_seq, uint8 frm_type, MI_FrameData_ptr& frmData);
    bool GetRecentData(stack<MI_FrameData_ptr>& frm_datas);
    bool OnStream(const StreamMediaFrameNotify& notify);

private:
    uint64 AdjustFrameSeq(uint32 frm_seq);
    CFrame_ptr GetFrame(uint32 frm_seq, uint8 frm_type);

private:
    boost::recursive_mutex lock_;
    SDeviceChannel dc_;
    string session_id_;

    map<uint64, CFrame_ptr> map_longseq2frame_;

    uint32 frm_total_cnt_;
    uint32 frm_invalid_msg_cnt_;
    uint32 frm_overflow_cnt_;
    uint32 frm_mallc_fail_cnt_;
    uint32 frm_recv_timeout_cnt_;

    tick_t last_recv_frm_tick_;
    tick_t last_recv_i_frm_tick_;
    tick_t last_recv_au_frm_tick_;

    uint32 last_gop_size_;
    uint32 last_i_frm_size_;
    uint64 last_frm_seq_;

};

typedef boost::shared_ptr<CFrameMgr> CFrameMgr_ptr;

#endif