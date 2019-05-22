#ifndef __DHCRYPTLIB_H__
#define __DHCRYPTLIB_H__

#include "typedef_win.h"

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

typedef struct
{   
	UINT wbuf[16];
	UINT hash[5];
	UINT count[2];
} SHA1_STATETYPE;

class CDHCryptLib  
{
public:
	CDHCryptLib();
	virtual ~CDHCryptLib();
	UINT BNMakeRandomNr(DWORD a[], UINT nSize);
	int BNMakePrime(DWORD p[],UINT nSize,PBYTE pEntropyPool=NULL, UINT nSizeEntropyPool=0);
	int BNIsPrime(DWORD W[],UINT nSize,UINT nrRounds);
	inline DWORD RandBetween(DWORD dwLower,DWORD dwUpper);
	DWORD MTRandom();
	BOOL MTInit(BYTE *pRandomPool=NULL, UINT nSize=0);

	int BNRabinMiller(const DWORD w[], UINT ndigits, UINT t);
	int BNGcd(DWORD g[], const DWORD x[], const DWORD y[], UINT nSize);
	int BNModInv(DWORD inv[], const DWORD u[], const DWORD v[], UINT nSize);
	int BNSquare(DWORD w[], const DWORD x[], UINT nSize);
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
	inline DWORD BNShiftLeft(DWORD a[], const DWORD *b, UINT x, UINT nSize);
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
protected:
	BOOL MTCollectEntropy(BYTE *pRandomPool, UINT nSize);
	void SHA1Hash(unsigned char *_pOutDigest, const unsigned char *_pData,UINT nSize);
	void SHA1_Start(SHA1_STATETYPE* _pcsha1);
	void SHA1_Finish(unsigned char* _pShaValue, SHA1_STATETYPE* _pcsha1);
	void SHA1_Hash(const unsigned char *_pData, unsigned int _iSize, SHA1_STATETYPE* _pcsha1);

	void SHA1_Transform(SHA1_STATETYPE* _pcsha1);

	static const UINT _SHA_MASK_[4];
	static const UINT _SHA_BITS_[4];

	static const DWORD SMALL_PRIMES[];
	static const UINT _NUMBEROFPRIMES_;

	UINT m_mtIndex;
	DWORD m_mtbuffer[624];
	BOOL m_bSeeded;

	int BNQhatTooBigHelper(DWORD qhat, DWORD  rhat,DWORD vn2, DWORD ujn2);
	DWORD BNMultSub(DWORD wn, DWORD w[], const DWORD v[], DWORD q, UINT n);
	void BNMultSubHelper(DWORD uu[2], DWORD qhat, DWORD v1, DWORD v0);
	int BNMultiplyHelper(DWORD p[2], const DWORD x, const DWORD y);
	DWORD BNDivideHelper(DWORD *q, DWORD *r, const DWORD u[2], DWORD v);
	int BNModSquareTmp(DWORD a[], const DWORD x[], DWORD m[], UINT nSize, DWORD temp[], DWORD tqq[], DWORD trr[]);
	int BNModuloTmp(DWORD r[], const DWORD u[], UINT nUSize, DWORD v[], UINT nVSize, DWORD tqq[], DWORD trr[]);
	int BNMultTmp(DWORD a[], const DWORD x[], const DWORD y[], DWORD m[], UINT nSize,  DWORD temp[], DWORD tqq[], DWORD trr[]);
};

#endif // __DHCRYPTLIB_H__
