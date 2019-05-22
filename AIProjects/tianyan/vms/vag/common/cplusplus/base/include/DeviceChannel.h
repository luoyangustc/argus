
#ifndef __SDEVICE_CHANNEL_H__
#define __SDEVICE_CHANNEL_H__
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "typedefine.h"
#include "typedef_win.h"
#include <boost/shared_ptr.hpp>
//#include "DeviceID.h"

struct SDeviceChannel
{
	//device_id_t did;
    string device_id_;
	uint16 channel_id_;
	uint8 stream_id_;
	SDeviceChannel()
    {
        memset(this,0,sizeof(*this));
    }

    SDeviceChannel( const string& device_id, uint16 ch_id, uint8 stream_id)
    {
        device_id_ = device_id;
        channel_id_ = ch_id;
        stream_id_ = stream_id;
    }

	const SDeviceChannel& operator=(const SDeviceChannel& right)
	{
        device_id_ = right.device_id_;
        channel_id_ = right.channel_id_;
        stream_id_ = right.stream_id_;
		return *this;
	}
	bool operator==(const SDeviceChannel& right)const
	{
		do 
		{
			if (device_id_ != right.device_id_)
			{
				break;
			}

			if (channel_id_ != right.channel_id_)
			{
				break;
			}

			if (stream_id_ != right.stream_id_)
			{
				break;
			}
			return true;
		} while (false);
		return false;
	}
	bool operator!=(const SDeviceChannel& right)const
	{
		return !(operator==(right));
	}
	bool operator<(const SDeviceChannel& right)const
	{
		do 
		{
			if (device_id_ < right.device_id_)
			{
				return true;
			}
			else if (device_id_ > right.device_id_)
			{
				break;
			}

			if (channel_id_<right.channel_id_)
			{
				return true;
			}
			else if (channel_id_>right.channel_id_)
			{
				break;
			}

			return stream_id_<right.stream_id_;
		} while (false);
		return false;
	}

	bool operator<=(const SDeviceChannel& right)const
	{
		if ( operator<(right) || operator ==(right) )
		{
			return true;
		}
		return false;
	}

	bool operator>(const SDeviceChannel& right)const
	{
		if ( operator<(right) || operator ==(right) )
		{
			return false;
		}
		return true;
	}

	bool operator>=(const SDeviceChannel& right)const
	{
		if ( operator<(right) )
		{
			return false;
		}
		return true;
	}

    string GetString() const
    {
        char szBuff[64]={0};
        sprintf_s(szBuff, sizeof(szBuff), "%s-%u-%u", device_id_.c_str(), channel_id_, stream_id_);
        return szBuff;
    }
};

typedef boost::shared_ptr<SDeviceChannel> SDeviceChannel_ptr;

#endif //__SDEVICE_CHANNEL_H__

