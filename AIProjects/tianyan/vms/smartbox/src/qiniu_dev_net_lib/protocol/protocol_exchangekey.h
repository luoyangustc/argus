#ifndef __PROTOCOL_EXCHANGEKEY_H__
#define __PROTOCOL_EXCHANGEKEY_H__

#include "vos_types.h"

#undef  EXT
#ifndef __PROTOCOL_EXCHANGEKEY_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

#define MAX_KEY_LENGTH  2048
#define DEVICE_1_KEY_POS  (2+4+4-1) //MsgHeader的msg_type的高字节

enum{
    EExchangekey_Nothing    = 0,
	EExchangekey_Rijndael   = 101,
	EExchangekey_Blowfish   = 102,
    EExchangekey_Device_01  = 1001,
};

#pragma pack (1)

typedef struct
{
	DWORD	Key[16];
	WORD	key_size;
}ExchangeKeyValue;

//ExchangeKeyRequest 0x01
typedef struct
{
	BYTE	key_P_length;
	BYTE	key_P[64];
	BYTE	key_A_length;
	BYTE	key_A[64];
}ExchangeKeyRequest_01;

typedef struct ExchangeKeyRequest
{
	DWORD	mask;
	//0X01
	ExchangeKeyRequest_01 keyPA_01;
	//0X02
	WORD	except_algorithm;
    vos_uint32_t	algorithm_param;
}ExchangeKeyRequest;

//ExchangeKeyResponse 0x01
typedef struct
{
	BYTE	key_B_length;
	BYTE	key_B[64];
	WORD	key_size;
}ExchangeKeyResponse_01;


typedef struct ExchangeKeyResponse
{
	DWORD mask;
    vos_int32_t resp_code;

	//0x01
    ExchangeKeyResponse_01 keyB_01;
	//0x02
    vos_uint16_t encry_algorithm;
    vos_uint32_t algorithm_param;
}ExchangeKeyResponse;
#pragma pack ()

EXT int Pack_MsgExchangeKeyRequest(char *buf, int buflen, ExchangeKeyRequest *req);
EXT int Unpack_MsgExchangeKeyResponse(char *pdata, vos_uint16_t datalen, ExchangeKeyResponse *resp);

#endif //__PROTOCOL_EXCHANGEKEY_H__

