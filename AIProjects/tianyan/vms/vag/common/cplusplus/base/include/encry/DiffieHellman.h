#ifndef __DIFFIEHELLMAN_H__
#define __DIFFIEHELLMAN_H__

#include "typedefine.h"
#include "typedef_win.h"
#include "DHCryptLib.h"

class DiffieHellman
{
public:
    DiffieHellman(UINT nSize);
    ~DiffieHellman();
public:
    int MakePrime();
    int ComputesA();
    UINT GetKeySize(){return key_size_;}
public:
    //for client
    int Get_A_P(OUT BYTE* pA, INOUT UINT* pA_Size, OUT BYTE* pP, INOUT UINT* pP_Size );
    int Set_B(IN const BYTE* pB, IN UINT B_Size);
    int ComputesS1();
    int Get_S1(OUT BYTE* pBuff, IN UINT buff_size);
public:
    //for server
    int Get_B(OUT BYTE* pB, INOUT UINT* pB_Size);
    int Set_A_P(IN const BYTE* pA, IN UINT A_Size, IN const BYTE* pP, IN UINT P_Size);
    int ComputesB();
    int ComputesS2();
    int Get_S2(OUT BYTE* pBuff, IN UINT buff_size);
public:
	CDHCryptLib cryptlib_;
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
    
};


#endif //__DIFFIEHELLMAN_H__

