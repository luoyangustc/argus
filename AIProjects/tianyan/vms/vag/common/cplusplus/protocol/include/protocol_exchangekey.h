/*************************************************************************
 Author: Lirunhua
 Created Time: 2014年08月13日 星期三 17时52分48秒
 File Name: protocol_exchangekey.h
 Description: 
 ************************************************************************/
#ifndef __PROTOCOL_EXCHANGEKEY_H__
#define __PROTOCOL_EXCHANGEKEY_H__
#include "protocol_header.h"
namespace protocol{

#define MAX_KEY_LENGTH  2048
#define DEVICE_1_KEY_POS  (2+4+4-1) //MsgHeader的msg_type的高字节

enum{
	EExchangekey_Nothing = 0,
	EExchangekey_Rijndael = 101,
	EExchangekey_Blowfish = 102,
	EExchangekey_Device_1 = 1001,
};

#pragma pack (push, 1)
struct ExchangeKeyRequest
{
	ExchangeKeyRequest()
	{
		memset(this,0,sizeof(*this));
	}
	uint32	mask;
	//0X01
	uint8	key_P_length;
	uint8	key_P[64];
	uint8	key_A_length;
	uint8	key_A[64];

	//0X02
	uint16	except_algorithm;
    uint32	algorithm_param;
};
#pragma pack (pop)
CDataStream& operator<<( CDataStream& _ds, ExchangeKeyRequest & _msgdata );
CDataStream& operator>>( CDataStream& _ds, ExchangeKeyRequest& _msgdata );

#pragma pack (push, 1)
struct ExchangeKeyResponse
{
	ExchangeKeyResponse()
	{
		memset(this,0,sizeof(*this));
	}

	uint32	mask;
    int32 resp_code;

    //0x01
	uint8	key_B_length;
	uint8	key_B[64];
	uint16	key_size;

	//0x02
	uint16	encry_algorithm;
    uint32	algorithm_param;
};
#pragma pack (pop)
CDataStream& operator<<( CDataStream& _ds, ExchangeKeyResponse & _msgdata );
CDataStream& operator>>( CDataStream& _ds, ExchangeKeyResponse& _msgdata );
};
#endif //__PROTOCOL_EXCHANGEKEY_H__

