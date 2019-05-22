#include "FrameMgr.h"
#include "to_string_util.h"

#define CACHE_FRAME_TIME_LEN (180)

CFrameMgr::CFrameMgr(SDeviceChannel dc)
    : dc_(dc)
{
    frm_total_cnt_ = 0;
    frm_invalid_msg_cnt_ = 0;
    frm_overflow_cnt_ = 0;
    frm_mallc_fail_cnt_ = 0;
    frm_recv_timeout_cnt_ = 0;

    last_recv_frm_tick_ = 0;
    last_recv_au_frm_tick_ = 0;
    last_recv_i_frm_tick_ = 0;

    last_gop_size_ = 0;
    last_i_frm_size_ = 0;
    last_frm_seq_ = 0;
}

CFrameMgr::~CFrameMgr()
{

}

bool CFrameMgr::IsActive()
{
    tick_t current_tick = get_current_tick();
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if((current_tick - last_recv_frm_tick_) > 6*1000)
    {
        return false;
    }
    return true;
}

bool CFrameMgr::IsAudioAlive()
{
    tick_t current_tick = get_current_tick();
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if((current_tick - last_recv_au_frm_tick_) > 6*1000)
    {
        return false;
    }
}

void CFrameMgr::Update()
{

}

CFrame_ptr CFrameMgr::GetFrame(uint32 frm_seq, uint8 frm_type)
{
    CFrame_ptr pFrame_ptr;
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        map<uint64, CFrame_ptr>::reverse_iterator it = map_longseq2frame_.rbegin();
        while( it!=map_longseq2frame_.rend() )
        {
            uint64 long_seq = it->first;
            uint32 low_seq = (uint32)long_seq;
            if( low_seq == frm_seq)
            {
                pFrame_ptr = it->second;
                break;
            }
        }

    } while (false);
    return pFrame_ptr;
}

bool CFrameMgr::GetFrameData(uint32 frm_seq, uint8 frm_type, MI_FrameData_ptr& frmData)
{
    CFrame_ptr pFrame = GetFrame(frm_seq, frm_type);
    if (!pFrame)
    {
        return false;
    }

    return pFrame->GetData(frmData);
}

bool CFrameMgr::GetRecentData(stack<MI_FrameData_ptr>& frm_datas)
{
    do 
    {
        map<uint64,CFrame_ptr> map_seq2frame;
        {
            boost::lock_guard<boost::recursive_mutex> lock(lock_);
            map_seq2frame = map_longseq2frame_;
        }

        MI_FrameData_ptr pFrmData;
        CFrame_ptr pFrame;
        int nCount = 0;
        map<uint64,CFrame_ptr>::reverse_iterator itLast = map_seq2frame.rbegin();
        for(; itLast != map_seq2frame.rend(); ++itLast)
        {
            pFrame = itLast->second;
            if (pFrame->GetData(pFrmData))
            {
                frm_datas.push(pFrmData);
            }
            else
            {
                if (!frm_datas.empty())
                {
                    while(!frm_datas.empty())
                    {
                        frm_datas.pop();
                    }
                    break;
                }
            }

            if (pFrame->IsIFrame())
            {
                break;
            }

            if (nCount++ > 100)
            {
                break;
            }
        }

    } while (false);
    return true;
}

bool CFrameMgr::OnStream(const StreamMediaFrameNotify& notify)
{
    tick_t curr_tick = get_current_tick();
    do
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        bool is_new_frm = false;
        uint64 long_seq = AdjustFrameSeq(notify.frame_seq);
        CFrame_ptr pFrame;
        map<uint64, CFrame_ptr>::iterator it = map_longseq2frame_.find(long_seq);
        if (it != map_longseq2frame_.end())
        {
            pFrame = it->second;
        }

        if (!pFrame)
        {
            pFrame = CFrame_ptr(new CFrame(notify));
            map_longseq2frame_[long_seq] = pFrame;
            is_new_frm = true;
        }

        if ( notify.mask&0x01 )
        {
            session_id_ = notify.session_id;
        }

        if ( !(notify.mask&0x08) )
        {
            frm_invalid_msg_cnt_++;
            pFrame->Discard();
            Error("dc(%s), this frame has no ts and size info!", dc_.GetString().c_str());
            break;
        }

        int ret = pFrame->SaveData(notify.offset, const_cast<uint8*>(notify.datas.data()),notify.data_size);
        if( ret == en_frm_sts_ok )
        {
            if( is_new_frm )
            {
                last_recv_frm_tick_ = curr_tick;
                if (notify.frame_type & 0x01)
                {
                    last_recv_i_frm_tick_ = curr_tick;
                }
                else if (notify.frame_type & 0x03)
                {
                    last_recv_au_frm_tick_ = curr_tick;
                }
                last_frm_seq_ = notify.frame_seq;
                frm_total_cnt_++;
            }

            if( pFrame->IsIFrame() && pFrame->IsFull() ) //收满一个I帧，清除之前的GOP
            {
                last_i_frm_size_ = pFrame->GetFrameSize();

                last_gop_size_ = 0; //clear
                map<uint64, CFrame_ptr>::iterator it = map_longseq2frame_.begin();
                while(it!=map_longseq2frame_.end())
                {
                    if( it->first < long_seq )
                    {
                        if( !it->second->IsAudio() )
                        {
                            last_gop_size_++;
                        }
                        map_longseq2frame_.erase(it++);
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        else
        {
            if( ret == en_frm_sts_offset_err ){
                frm_invalid_msg_cnt_++;
            }else if( ret == en_frm_sts_overflow ){
                frm_overflow_cnt_++ ;
            }else if( ret == en_frm_sts_crc_err ){
                frm_invalid_msg_cnt_++;
            }else if( ret == en_frm_sts_mallc_fail ){
                frm_mallc_fail_cnt_++;
            }else if( ret == en_frm_sts_timeout ){
                frm_recv_timeout_cnt_++;
            }

            pFrame->Discard();
            Error("session_id(%s), dc(%s), failed ret(%d)!", notify.session_id.c_str(), dc_.GetString().c_str(), ret);
            break;
        }

        return true;
    } while (false);
    return false;
}

uint64 CFrameMgr::AdjustFrameSeq( uint32 frm_seq )
{
    uint64 long_seq = 0;

    if ( !map_longseq2frame_.empty() )
    {
        uint64 last_longseq = map_longseq2frame_.rbegin()->first;
        uint32 last_low_seq = (uint32)(last_longseq);
        uint32 last_high_seq = (uint32)(last_longseq>>32);
        if ( frm_seq == last_low_seq )
        {
            long_seq = last_longseq;
        }
        else if ( frm_seq > last_low_seq )
        {
            long_seq = ( ((uint64)last_high_seq) << 32 ) | (uint64)frm_seq;
        }
        else
        {
            uint32 delta = last_low_seq - frm_seq;
            if ( delta > 125 ) // 差值大于125，认为seq翻转
            {
                last_high_seq++; // 高32位溢出暂不考虑
                long_seq = ( ((uint64)last_high_seq) << 32 ) | (uint64)frm_seq;
                Warn("session_id(%s), dc(%s), frame seq reverse(hight_seq:%u, low_seq:%u, long_seq:%llu)!", 
                    session_id_.c_str(), dc_.GetString().c_str(), last_high_seq, frm_seq, long_seq);
            }
            else
            {
                long_seq = ( ((uint64)last_high_seq) << 32 ) | (uint64)frm_seq;
            }
        }
    }
    else
    {
        long_seq = (uint64)frm_seq;
    }

    return long_seq;
}

void CFrameMgr::DumpInfo(Variant& info)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    info["expire"] = calc_time_unit((uint32)(get_current_tick() - last_recv_frm_tick_)/1000);
    info["recv_total_cnt"] = frm_total_cnt_;
    info["invalid_msg_cnt"] = frm_invalid_msg_cnt_;
    info["overflow_cnt"] = frm_overflow_cnt_;
    info["malloc_fail_cnt"] = frm_mallc_fail_cnt_;
    info["recv_timeout_cnt"] = frm_recv_timeout_cnt_;
    info["last_recv_frm_tick"] = last_recv_frm_tick_;
    info["last_recv_i_frm_tick"] = last_recv_i_frm_tick_;
    info["last_recv_au_frm_tick"] = last_recv_au_frm_tick_;
    info["last_gop_size"] = last_gop_size_;
    info["last_i_frm_size"] = last_i_frm_size_;
    info["last_frm_seq"] = (uint32)last_frm_seq_;
}