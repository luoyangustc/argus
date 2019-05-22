#ifndef __CDIFFIE_HELL_MAN_H__
#define __CDIFFIE_HELL_MAN_H__

#include "cdh_crypt_lib.h"
#include "protocol_exchangekey.h"

#undef  EXT
#ifndef __CDIFFIE_HELL_MAN_C__
#define  EXT extern
#else
#define  EXT 
#endif

#define MAX_SOCET_FD 4
#pragma pack(1)
typedef struct
{
	UINT	size_;
	UINT	key_size_;
	DWORD	p_[16];
	DWORD	g_[16];
	DWORD	A_[16];
	DWORD	B_[16];
	DWORD	a_[16];
	DWORD	b_[16];
	DWORD	S1_[16];
	DWORD	S2_[16];
	DHCryptData dh_crypt_data;
	int   	socket_fd;
	UINT    use_key_size_;
	BOOL    connecting;
}DiffieHellManData;

typedef struct
{
	DWORD connect_num;
	DiffieHellManData _data[MAX_SOCET_FD];
}DiffieHellManDataList;

#pragma pack()

EXT int InitDiffieHellman(UINT nSize, int socket_fd);
EXT void InitDHDataList();
EXT int MakePrime(DWORD socket_fd);
EXT int ComputesA(DWORD socket_fd); 
EXT int ComputesB(DWORD socket_fd); 
EXT int ComputesS1(DWORD socket_fd); 
EXT int ComputesS2(DWORD socket_fd); 

EXT int Printf_a_p(ExchangeKeyRequest *pMsg, DWORD socket_fd);
EXT void Set_B(ExchangeKeyResponse *pMsg, DWORD socket_fd);
EXT int Printf_S1(DWORD socket_fd,DWORD *pdw_Buf,int buf_len);
EXT int Get_size_(DWORD socket_fd);
EXT int Get_exchangekey_(int socket_fd,char *pS_);
EXT int Clear_DH_conn_status(DWORD socket_fd);

#endif //__DIFFIEHELLMAN_H__
