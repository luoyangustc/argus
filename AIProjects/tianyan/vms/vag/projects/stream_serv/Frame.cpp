#include "Frame.h"

const int MAX_FRAME_SIZE = (1024*1024);

CFrame::CFrame(const StreamMediaFrameNotify& notify)
    : status_(en_frm_sts_ok)
    , frame_type_(notify.frame_type)
    , frame_av_seq_(notify.frame_av_seq)
    , frame_seq_(notify.frame_seq)
    , frame_base_time_(notify.frame_base_time)
    , frame_ts_(notify.frame_ts)
    , frame_size_(notify.frame_size)
{
    add_tick_ = get_current_tick();
    recv_size_ = 0;

    if (notify.mask & 0x04)
    {
        crc32_hash_ = notify.crc32_hash;
    }
    else
    {
        crc32_hash_ = 0;
    }

    do 
    {
        if(frame_size_ > MAX_FRAME_SIZE)
        {
            status_ = en_frm_sts_overflow;
            break;
        }

        data_ = boost::shared_array<uint8>(new(std::nothrow) uint8[frame_size_]);
        if (!data_)
        {
            status_ = en_frm_sts_mallc_fail;
            break;
        }

        memset(data_.get(), 0, frame_size_);
        int piece_cnt = (frame_size_ + MAX_MSG_BUFF_SIZE - 1)/MAX_MSG_BUFF_SIZE;
        m_bf_.SetFieldSize(piece_cnt);
        m_bf_.init(FALSE);
    } while (false);
}

CFrame::~CFrame()
{

}

int CFrame::GetStatus()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    if (status_ != en_frm_sts_ok)
    {
        return status_;
    }

    if ( !IsFull() && (get_current_tick() - add_tick_ > 20*1000) )
    {
        status_ = en_frm_sts_timeout;
        return en_frm_sts_timeout;
    }

    return en_frm_sts_ok;
}

void CFrame::Discard()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    status_ = en_frm_sts_force_discard;
}

bool CFrame::IsFull()
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (m_bf_.IsFull())
        {
            return true;
        }
    } while (false);
    return false;
}

int CFrame::SaveData(uint32 offset, PBYTE buffer, uint32 data_size)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if ( (status_ != en_frm_sts_ok) < 0)
        {
            Error("status(%d) is incorrect!", status_);
            break;
        }

        if (offset%MAX_MSG_BUFF_SIZE != 0)
        {
            status_ = en_frm_sts_offset_err;
            Error("offset(%u) is incorrect, frame_size(%u)!", offset, frame_size_);
            break;
        }

        if ( offset+data_size > frame_size_)
        {
            status_ = en_frm_sts_overflow;
            Error("offset(%u)+length(%u)>frame_size(%u). frame is overflow!", offset, data_size, frame_size_);
            break;
        }

        int bit = offset/MAX_MSG_BUFF_SIZE;
        recv_size_ += data_size;
        memcpy(data_.get() + offset, buffer, data_size);
        m_bf_.SetBitValue(bit,1);

        if (m_bf_.IsFull())
        {
            if (crc32_hash_)
            {
                uint32 calc_crc = calc_crc32 (data_.get(), frame_size_);
                if (calc_crc != crc32_hash_)
                {
                    status_ = en_frm_sts_crc_err;
                    m_bf_.init(FALSE);
                    Error("Frame crc32 ERROR:seq(%u),ts:(%u),size(%u)", frame_seq_, frame_ts_, frame_size_);
                    break;
                }	
            }
        }
    } while (false);
    return status_;
}

bool CFrame::GetData(MI_FrameData_ptr& pFrameData)
{
    do 
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        if (!m_bf_.IsFull())
        {
            break;
        }

        if (GetStatus() < 0)
        {
            break;
        }

        pFrameData = MI_FrameData_ptr(new MI_FrameData());
        pFrameData->data_ = data_;
        pFrameData->is_audio_ = IsAudioFrame(frame_type_);
        pFrameData->is_i_frame_ = IsKeyFrame(frame_type_);
        pFrameData->is_frist_ = false;
        pFrameData->frame_type_ = frame_type_;
        pFrameData->frame_seq_ = frame_seq_;
        pFrameData->frame_base_time_ = frame_base_time_;
        pFrameData->frame_ts_ = frame_ts_;
        pFrameData->frame_size_ = frame_size_;
        pFrameData->frame_av_seq_ = frame_av_seq_;
        pFrameData->crc32_hash_ = crc32_hash_;
        return true;
    } while (false);
    return false;
}
