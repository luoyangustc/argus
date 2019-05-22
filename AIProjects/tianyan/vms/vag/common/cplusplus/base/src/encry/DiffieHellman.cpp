#include "DiffieHellman.h"
#include <string.h>

DiffieHellman::DiffieHellman(UINT nSize)
	:size_(nSize + 1)
	,key_size_(nSize)
{
	memset(p_,0,sizeof(p_));
	memset(g_,0,sizeof(g_));
	memset(A_,0,sizeof(A_));
	memset(B_,0,sizeof(B_));
	memset(a_,0,sizeof(a_));
	memset(b_,0,sizeof(b_));
	memset(S1_,0,sizeof(S1_));
	memset(S2_,0,sizeof(S2_));

	cryptlib_.BNSetEqualdw(g_, 5, key_size_);
}

DiffieHellman::~DiffieHellman()
{
}

int DiffieHellman::MakePrime()
{
	return cryptlib_.BNMakePrime(p_,size_);
}

int DiffieHellman::ComputesA()
{
	for (UINT j = 0; j <key_size_ ; j++)
		a_[j] = cryptlib_.MTRandom();	
	return cryptlib_.BNModExp(A_, g_, a_, p_,key_size_);
}
int DiffieHellman::ComputesB()
{
	for (UINT j = 0; j <key_size_ ; j++)
		b_[j] = cryptlib_.MTRandom();	
	return cryptlib_.BNModExp(B_, g_, b_, p_,key_size_);
}
int DiffieHellman::ComputesS1()
{
	return cryptlib_.BNModExp(S1_, B_, a_, p_,key_size_);
}
int DiffieHellman::ComputesS2()
{
	return cryptlib_.BNModExp(S2_, A_, b_, p_,key_size_);
}

int DiffieHellman::Get_A_P(OUT BYTE* pA, INOUT UINT* pA_Size, OUT BYTE* pP, INOUT UINT* pP_Size )
{
    if( !pA || (*pA_Size < key_size_*4) )
    {
        return -1;
    }

    if( !pP || (*pP_Size < key_size_*4) )
    {
        return -2;
    }

    memcpy(pA, A_, key_size_*4);
    *pA_Size = key_size_*4;

    memcpy(pP, p_, size_*4);
    *pP_Size = size_*4;

    return 0;
}

int DiffieHellman::Set_A_P(IN const BYTE* pA, IN UINT A_Size, IN const BYTE* pP, IN UINT P_Size)
{
    if( (!pA) || (A_Size != key_size_*4) )
    {
        return -1;
    }

    if( (!pP) || (P_Size != size_*4) )
    {
        return -2;
    }

    memcpy(A_, pA, A_Size);
    memcpy(p_, pP, P_Size);

    return 0;
}

int DiffieHellman::Get_B(OUT BYTE* pB, INOUT UINT* pB_Size)
{
    if( !pB || (*pB_Size < key_size_*4) )
    {
        return -1;
    }


    memcpy(pB, B_, key_size_*4);
    *pB_Size = key_size_*4;

    return 0;
}

int DiffieHellman::Set_B(IN const BYTE* pB, IN UINT B_Size)
{
    if( (!pB) || (B_Size != key_size_*4 ) )
    {
        return -1;
    }

    memcpy(B_, pB, B_Size);

    return 0;
}

int DiffieHellman::Get_S1(OUT BYTE* pBuff, IN UINT buff_size)
{
    if( !pBuff || (buff_size > (key_size_*4)) )
    {
        return -1;
    }

    memcpy(pBuff, S1_, buff_size);

    return buff_size;
}

int DiffieHellman::Get_S2(OUT BYTE* pBuff, IN UINT buff_size)
{
    if( !pBuff || ( buff_size > (key_size_*4)) )
    {
        return -1;
    }

    memcpy(pBuff, S2_, buff_size);

    return buff_size;
}