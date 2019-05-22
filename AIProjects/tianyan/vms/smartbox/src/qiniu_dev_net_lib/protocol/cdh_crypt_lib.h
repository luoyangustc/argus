#ifndef __CDH_CRYPT_LIB_H__
#define __CDH_CRYPT_LIB_H__

#include "vos_types.h"

#if ((0x1234 >> 24) == 1)
#define  SYSTEM_LITTLE_ENDIAN 1234
#elif ((0x4321 >> 24) == 1)
#define SYSTEM_BIG_ENDIAN      4321
#endif

#define LOHALF(x) ((DWORD)((x) & _MAXHALFNR_))
#define HIHALF(x) ((DWORD)((x) >> sizeof(DWORD)*4 & _MAXHALFNR_))
#define TOHIGH(x) ((DWORD)((x) << sizeof(DWORD)*4))
#ifdef _WINDOWS
#define rotate32(x,n) _lrotl((x), (n))
#else
#define rotate32(x,n) (((x) << n) | ((x) >> (32 - n)))
#endif

#if (SYSTEM_BIG_ENDIAN)

#define SHA_BLOCK32(x) (x)
#define _HIBITMASK_ 0x00000008
#define _MAXIMUMNR_ 0xffffffff
#define _MAXHALFNR_ 0x000Lffff 
#else
#define SHA_BLOCK32(x) ((rotate32((x), 8) & 0x00ff00ff) | (rotate32((x), 24) & 0xff00ff00))
#define _HIBITMASK_ 0x80000000
#define _MAXIMUMNR_ 0xffffffff
#define _MAXHALFNR_ 0xffffL 
#endif

#define SHA1_BLOCK_SIZE  64
#define SHA1_DIGEST_SIZE 20

#define F0to19(x,y,z)       (((x) & (y)) ^ (~(x) & (z)))
#define F20to39(x,y,z)		((x) ^ (y) ^ (z))
#define F40to59(x,y,z)      (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define F60to79(x,y,z)		 F20to39(x,y,z)

#define sha_round(func,k)  t = a; a = rotate32(a,5) + func(b,c,d) + e + k + w[i];\
	e = d;d = c; c = rotate32(b, 30); b = t;
#pragma pack(1)

typedef struct
{   
	UINT wbuf[16];
	UINT hash[5];
	UINT count[2];
} SHA1_STATETYPE;

typedef struct
{
	UINT m_mtIndex;
	DWORD m_mtbuffer[624];
	BOOL m_bSeeded;
}DHCryptData;
#pragma pack()

void InitCDHCryptLib(DHCryptData *pData);
UINT BNMakeRandomNr(DWORD a[], UINT nSize,DHCryptData *pData);
int BNMakePrime(DWORD p[],UINT nSize,PBYTE pEntropyPool, UINT nSizeEntropyPool,DHCryptData *pData);
int BNIsPrime(DWORD W[],UINT nSize,UINT nrRounds,DHCryptData *pData);

VOS_INLINE DWORD RandBetween(DWORD dwLower,DWORD dwUpper,DHCryptData *pData);

DWORD MTRandom(DHCryptData *pData);
BOOL MTInit(BYTE *pRandomPool, UINT nSize,DHCryptData *pData);

int BNRabinMiller(const DWORD w[], UINT ndigits, UINT t,DHCryptData *pData);
int BNGcd(DWORD g[], const DWORD x[], const DWORD y[], UINT nSize);
int BNModInv(DWORD inv[], const DWORD u[], const DWORD v[], UINT nSize);

VOS_INLINE int BNSquare(DWORD w[], const DWORD x[], UINT nSize);

int BNModExp(DWORD yout[], const DWORD x[], const DWORD e[], const DWORD m[], UINT nSize);
DWORD BNModMult(DWORD a[], const DWORD x[], const DWORD y[], const DWORD m[], UINT nSize);
DWORD BNMod(DWORD r[], const DWORD u[], UINT nUSize, DWORD v[], UINT nVSize);
UINT BNUiceil(double x);
DWORD BNModdw(DWORD a[],DWORD d, UINT nSize);
void BNFree(DWORD **p);
DWORD * BNAlloc(UINT nSize);
UINT BNBitLength(const DWORD *d,UINT nSize);
DWORD BNSubtractdw(DWORD w[], const DWORD u[], DWORD v,  UINT  nSize);
int BNComparedw(const DWORD a[], DWORD b, UINT nSize);
int BNCompare(const DWORD a[], const DWORD b[], UINT nSize);
DWORD BNShiftRight(DWORD a[], const DWORD *b, DWORD x, DWORD nSize);
VOS_INLINE DWORD BNShiftLeft(DWORD a[], const DWORD *b, UINT x, UINT nSize);

DWORD BNDividedw(DWORD q[], const DWORD u[], DWORD  v, UINT nSize);
void BNSetEqualdw(DWORD a[], const DWORD d, UINT nSize);
void BNSetEqual(DWORD a[], const DWORD b[], UINT nSize);
int BNIsZero(const DWORD a[], UINT nSize);
int BNIsEqual(const DWORD a[], const DWORD b[], UINT nSize);
UINT BNSizeof(const DWORD A[], UINT nSize);
void BNSetZero(DWORD A[],UINT nSize);
int BNDivide(DWORD q[], DWORD r[], const DWORD u[], UINT usize,DWORD v[],UINT vsize);
DWORD BNSubtract(DWORD C[], const DWORD A[], const DWORD B[], const UINT nSize);
DWORD BNAdd(DWORD C[], const DWORD A[],const  DWORD B[], const UINT nSize);
DWORD BNAdddw(DWORD w[], const DWORD u[], DWORD v, UINT nSize);
DWORD BNMultiply(DWORD C[], const DWORD A[], const DWORD B[], const UINT nSize);
DWORD BNMultiplydw(DWORD w[], const DWORD u[], DWORD v, UINT nSize);

#endif // __DHCRYPTLIB_H__
