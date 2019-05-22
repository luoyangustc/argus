
#include "protocol_cloudstorage.h"
namespace protocol{

    CDataStream& operator<<(CDataStream& ds, CSTTSDataPushReq& req)
    {
        ds << req.mask;
        if( req.mask & 0x01 )
        {
            ds.writestring(req.device_id.c_str());
            ds << req.channel_idx;
            ds << req.rate_type;
            ds << req.storage_type;
            ds << req.ts_timestamp;
            ds << req.ts_total_size;
        }
        if( req.mask & 0x02 )
        {
            ds << req.ts_offset;
            ds << req.data_size;
            for (int i = 0; i< req.data_size; ++i)
            {
                ds << req.datas[i];
            }
        }
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CSTTSDataPushReq& req)
    {
        ds >> req.mask;
        if( req.mask & 0x01 )
        {
            req.device_id = ds.readstring();
            ds >> req.channel_idx;
            ds >> req.rate_type;
            ds >> req.storage_type;
            ds >> req.ts_timestamp;
            ds >> req.ts_total_size;
        }
        if( req.mask & 0x02 )
        {
            ds >> req.ts_offset;
            ds >> req.data_size;
            for (int i = 0; i< req.data_size; ++i)
            {
                uint8 data;
                ds >> data;
                req.datas.push_back(data);
            }
        }
        return ds;
    }

    CDataStream& operator<<(CDataStream& ds, CSTTSDataPushResp& resp)
    {
        ds << resp.mask;
        ds << resp.resp_code;
        return ds;
    }
    CDataStream& operator>>(CDataStream& ds, CSTTSDataPushResp& resp)
    {
        ds >> resp.mask;
        ds >> resp.resp_code;
        return ds;
    }
}

