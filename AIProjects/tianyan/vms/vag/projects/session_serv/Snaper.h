#ifndef __SNPER_H__
#define __SNPER_H__

#include <boost/shared_array.hpp>
#include "typedefine.h"
#include "tick.h"
#include "protocol_device.h"

using namespace protocol;

class CSnaper
{
public:
    CSnaper();
    ~CSnaper();
    bool OnDeviceSnaperResp(const DeviceSnapResp& resp);
    bool IsFull();
    int GetStatus();
    uint32 life_time(){return (uint32)(get_current_tick()-last_active_tick_);}
private:
    string gen_pic_name();
    string gen_pic_timestamp();
public:
    bool init_flag_;
    int status_;
    tick_t last_active_tick_;
    string pic_name_;

    string device_id_;
    uint16 channel_id_;
    string pic_fmt_;
    uint32 pic_size_;

    uint32 recv_data_size_;
    boost::shared_array<uint8> datas_;
};

typedef boost::shared_ptr<CSnaper> CSnaper_ptr;

#endif //__SNPER_H__