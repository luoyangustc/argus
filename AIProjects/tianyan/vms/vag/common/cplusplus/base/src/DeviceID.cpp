#include <stdlib.h>
#include <stdio.h>

#include "DeviceID.h"
#include "ZBase64.h"
#include "encry/crc32.h"
#include "encry/crc8.h"
#include "encry/crc16.h"

void CDeviceID::dump_info()
{
	printf("len:%u,",data_len_);

	for (int i = 0; i< DEVICE_MAX_LEN; ++i)
	{
		printf("%2x,",(uint32)data_[i]);
	}

	printf("\n");	
}

const CDeviceID & CDeviceID::operator=(const char * szValue)
{
	do 
	{
		clear();
#ifndef _WINDOWS
		int len	= lstrlen((LPCSTR)szValue);
#else		
		int  len = strlen(szValue);
#endif
		if (len%4)
		{
			break;
		}

		data_len_ = ZBase64::DecodeLength(len);
		if( data_len_ == 0 || data_len_>DEVICE_MAX_LEN )
		{	
			break;
		}

		int out_len = data_len_;
		ZBase64::Decode(szValue,len,data_,out_len);
		data_len_ = out_len;
	} while (false);
	return *this;	
}

CDeviceID&	CDeviceID::operator=(const CDeviceID& right)
{
	clear();
	if( right.data_len_ && right.data_len_ <= DEVICE_MAX_LEN)
	{
		data_len_ = right.data_len_;
		memcpy(data_,right.data_,data_len_);	
	}
	return * this;
}

bool CDeviceID::operator==(const CDeviceID & first) const
{
    return memcmp(this,&first,sizeof(CDeviceID))==0?true:false;
}
bool CDeviceID::operator != (const CDeviceID & first) const
{
	return memcmp(this,&first,sizeof(CDeviceID))!=0?true:false;	
}
bool CDeviceID::operator >= (const CDeviceID & first) const
{
	return memcmp(this,&first,sizeof(CDeviceID))>=0?true:false;	
}
bool CDeviceID::operator <= (const CDeviceID & first) const
{
	return memcmp(this,&first,sizeof(CDeviceID))<=0?true:false;
}
bool CDeviceID::operator < (const CDeviceID & first) const
{
	return memcmp(this,&first,sizeof(CDeviceID))<0?true:false;	
}
bool CDeviceID::operator > (const CDeviceID & first) const
{
	return memcmp(this,&first,sizeof(CDeviceID))>0?true:false;	
}

CDeviceID::CDeviceID(const device_id_t& did)
{
	do 
	{
		clear();
		if (did.device_id_length > DEVICE_MAX_LEN)
		{
			break;
		}

		data_len_ = did.device_id_length;
		int mod = data_len_ % 3;
		int offset = 0;
		if (mod)
		{
			offset = (3-mod);
		}
		
		memcpy(data_+offset, did.device_id, data_len_);
		data_len_ += offset;	
	} while (false);
}

CDeviceID::CDeviceID(const char * szValue)
{
	clear();
	this->operator=(szValue);
}
CDeviceID::CDeviceID(const CDeviceID& right)
{
	clear();
	this->operator = (right);
}

int CDeviceID::getIndex(uint8 group_num)
{
	if (data_len_>DEVICE_MAX_LEN || data_len_==0)
	{
		return 0;
	}

	uint64 sum = 0;
	for (int i = 0; i<data_len_; ++i)
	{
		sum += data_[i];
	}

	return sum % group_num;	
}

bool CDeviceID::is_valid()
{
	do 
	{
		if (data_len_ == 0 || data_len_%3)
		{
			break;
		}

		//最老的新设备校验方法
		do
		{
			unsigned int hash = calc_crc32 (data_, data_len_-3);
			int h_hash = ntohl(hash);
			unsigned char * pHash = (unsigned char *)&h_hash;

			if (data_[data_len_-3] != (pHash[2]& 0x7D) )
			{
				break;
			}

			if (data_[data_len_-2] != (pHash[1]&0XF7))
			{
				break;
			}

			if (data_[data_len_-1] != (pHash[3]&0XDF) )
			{
				break;
			}

			return true;
		}while(false);

		if (data_len_ == 15 )
		{
			//只校验了前13个字节的校验方法
			do 
			{
				uint16 hash = CRC16_1 (data_, 13);
				uint16 h_hash = ntohs(hash);
				unsigned char * pHash = (unsigned char *)&h_hash;

				if ( (data_[13]&0X0F) != (pHash[0]&0x07) )
				{
					break;
				}

				if (data_[14] != (pHash[1]&0XDF))
				{
					break;
				}
				return true;
			} while (false);
			
			//校验了前14个字节的校验方法
			do 
			{
				uint8 tmp_buff[15];
				memcpy(tmp_buff,data_,15);
				uint8 orig_data_13 = tmp_buff[13];
				tmp_buff[13] = orig_data_13&0xF0;

				uint16 hash = CRC16_1 (tmp_buff, 14);
				uint16 h_hash = ntohs(hash);
				unsigned char * pHash = (unsigned char *)&h_hash;

				if ( (orig_data_13&0X0F) != (pHash[0]&0x07) )
				{
					break;
				}

				if (tmp_buff[14] != (pHash[1]&0XDF))
				{
					break;
				}
				return true;
			} while (false);
		}while(false);
	} while (false);
	return false;
}

bool CDeviceID::generate_device_id(const string& orig_did,string& out_new_did)
{
	do 
	{

		int len	= orig_did.length();
		if (len == 18)
		{
#if 1
			{
				bool is_valid_orig_str = false;
				char c_begin = *orig_did.begin();
				if ( c_begin >= 'A' && c_begin <= 'Z' )
				{
					is_valid_orig_str = true;
				}

				if ( c_begin >= 'a' && c_begin <= 'z' )
				{
					is_valid_orig_str = true;
				}
				if (is_valid_orig_str==false)
				{
					break;
				}				
			}

			int out_len = 15;
			boost::shared_array<unsigned char> pData(new unsigned char[15]);
			memset(pData.get(),0,15);
			string orig_base64_did = orig_did + "==";
			ZBase64::Decode(orig_base64_did.c_str(),20,pData.get(),out_len);	

			uint16 hash = CRC16_1(pData.get(), 14);
			uint16 n_hash = htons(hash);
			unsigned char * pHash = (unsigned char *)&n_hash;

			(pData.get())[13] = (((pData.get())[13])&0XF0) | (pHash[0]&0x07);
			(pData.get())[14] = pHash[1]&0xDF;
			out_new_did = ZBase64::Encode(pData.get(),15);
			return true;		
#else
			//只根据13个字节生成校验码，不再推荐
			{
				char c_begin = *orig_did.begin();
				if ( c_begin < 'A' || c_begin > 'Z')
				{
					break;
				}
			}

			int out_len = 15;
			boost::shared_array<unsigned char> pData(new unsigned char[15]);
			memset(pData.get(),0,15);
			string orig_base64_did = orig_did + "==";
			ZBase64::Decode(orig_base64_did.c_str(),20,pData.get(),out_len);	

			uint16 hash = CRC16_1(pData.get(), 13);
			uint16 n_hash = htons(hash);
			unsigned char * pHash = (unsigned char *)&n_hash;

			(pData.get())[13] = (((pData.get())[13])&0XF0) | (pHash[0]&0x07);
			(pData.get())[14] = pHash[1]&0xDF;
			out_new_did = ZBase64::Encode(pData.get(),15);
			return true;
#endif //
		}		
	} while (false);
	return false;
}


bool  CDeviceID::get_raw_device_id(device_id_t& raw_did)
{
 	do
 	{
 		//the buffer max length in device_id max size is 21
 		if(data_len_ == 0 || data_len_ >21)
 		{
 			break;
 		}
		
		memcpy(raw_did.device_id, data_, data_len_);
		raw_did.device_id_length = data_len_;
		return true;
 	}while(false);
	return false;
}


int CDeviceID::get_group(uint32 group_num)
{
	do 
	{
		if (data_len_>DEVICE_MAX_LEN || data_len_==0)
		{
			break;
		}

		uint64 sum = 0;
		for (int i = 0; i<data_len_; ++i)
		{
			sum += data_[i];
		}

		return sum % group_num;	

	} while (false);
	return -1;	
}

void CDeviceID::getidstring(string & strSha1) const
{
	if ( data_len_>DEVICE_MAX_LEN || data_len_==0 )
	{
		return;
	}

	strSha1 = ZBase64::Encode(data_,data_len_);
}
