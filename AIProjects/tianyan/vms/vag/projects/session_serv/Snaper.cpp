#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "Snaper.h"

CSnaper::CSnaper()
{
    init_flag_ = false;
    status_ = 0;
    last_active_tick_ = get_current_tick();

    recv_data_size_ = 0;
}

CSnaper::~CSnaper()
{

}

bool CSnaper::OnDeviceSnaperResp(const DeviceSnapResp& resp)
{
    if( status_ < 0 )
    {
        return false;
    }

    if( !init_flag_ )
    {
        if ( resp.mask&0x01 )
        {
            device_id_ = resp.device_id;
            channel_id_ = resp.channel_id;
            pic_fmt_ = resp.pic_fmt;
            pic_size_ = resp.pic_size;

            datas_ = boost::shared_array<uint8>(new(std::nothrow) uint8[pic_size_]);
            if ( !datas_ )
            {
                status_ = -1;
                return false;
            }

            {
                pic_name_ = gen_pic_name();
                if( pic_name_.empty() )
                {
                    status_ = -6;
                    return false;
                }
            }

            init_flag_ = true;
        }
        else
        {
            status_ = -2;
            return false;
        }
    }

    if ( resp.mask&0x02 )
    {
        if ( resp.offset != recv_data_size_ )
        {
            status_ = -3;
            return false;
        }

        if ( resp.data_size > pic_size_ )
        {
            status_ = -4;
            return false;
        }

        memcpy(datas_.get()+recv_data_size_, resp.datas.data(), resp.data_size);
        recv_data_size_ += resp.data_size;
    }
    else
    {
        status_ = -5;
        return false;
    }

    last_active_tick_ = get_current_tick();
    return true;
}


int CSnaper::GetStatus()
{
    return status_;
}

bool CSnaper::IsFull()
{
    if ( recv_data_size_ == pic_size_ )
    {
        return true;
    }

    return false;
}

string CSnaper::gen_pic_name()
{
    string timestamp = gen_pic_timestamp();
    if( timestamp.empty() )
    {
        return "";
    }

    char szbuff[256] = {0};
    int ret = snprintf(szbuff, sizeof(szbuff) -1, "%s_%d_%s.%s", 
                        device_id_.c_str(), 
                        (int)channel_id_, 
                        timestamp.c_str(),
                        pic_fmt_.c_str() );
    if( ret < 0 )
    {
        return "";
    }
    szbuff[ret] = '\0';

    return szbuff;
}

string CSnaper::gen_pic_timestamp()
{
    char szbuff[64] = {0};
    struct timeval time_value;
    gettimeofday(&time_value, NULL);

    
    char* pos = szbuff;
    size_t len = strftime(pos, sizeof(szbuff), "%Y_%m_%d_%H_%M_%S", localtime((time_t*)&time_value.tv_sec));
    if( len == 0)
    {
        return "";
    }
    int ret = snprintf(pos + len, sizeof(szbuff) - len - 1,  "_%llu", (long long unsigned int)time_value.tv_usec);
    if( ret < 0 )
    {
        return "";
    }
    *(pos + len + ret) = '\0';

    return szbuff;
}