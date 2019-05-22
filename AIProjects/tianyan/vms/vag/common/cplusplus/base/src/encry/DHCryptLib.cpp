#include "DHCryptLib.h"
#include <sys/types.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#else
#ifdef ANDROID
#include <linux/kernel.h>
#else
#ifndef _WINDOWS
#include <sys/sysinfo.h>
#endif
#endif
#endif

#ifndef _WINDOWS
#include <unistd.h>
#include <sys/syscall.h>
#include "pthread.h"
#endif
//#include "GetTickCount.h"
#include <string.h>

#include <stdlib.h>

#ifndef _WINDOWS
pid_t gettid()
{
#ifdef ANDROID
    return syscall(__NR_gettid/*SYS_gettid*/);
#else
#ifndef _WINDOWS
	return syscall(SYS_gettid);
#else
	return ::GetCurrentThreadId();
#endif
#endif
}
#endif //_WINDOWS

#undef min
#undef max
#define min(x,y)	 ((x)<(y)?(x):(y))
#define max(x,y)	((x)>(y)?(x):(y))

CDHCryptLib::CDHCryptLib()
{
	m_mtIndex=0;
	m_bSeeded=FALSE;
}

CDHCryptLib::~CDHCryptLib()
{

}

const DWORD CDHCryptLib::SMALL_PRIMES[] =  {
	3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 
	47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 
	103, 107, 109, 113,127, 131, 137, 139, 149, 151, 
	157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 
	211, 223, 227, 229,233, 239, 241, 251, 257, 263, 
	269, 271, 277, 281,283, 293, 307, 311, 313, 317, 
	331, 337, 347, 349,353, 359, 367, 373, 379, 383, 
	389, 397, 401, 409,419, 421, 431, 433, 439, 443, 
	449, 457, 461, 463,467, 479, 487, 491, 499, 503, 
	509, 521, 523, 541,547, 557, 563, 569, 571, 577, 
	587, 593, 599, 601,607, 613, 617, 619, 631, 641, 
	643, 647, 653, 659,661, 673, 677, 683, 691, 701, 
	709, 719, 727, 733,739, 743, 751, 757, 761, 769, 
	773, 787, 797, 809,811, 821, 823, 827, 829, 839, 
	853, 857, 859, 863,877, 881, 883, 887, 907, 911, 
	919, 929, 937, 941,947, 953, 967, 971, 977, 983, 
	991, 997,
};

const UINT CDHCryptLib::_NUMBEROFPRIMES_=sizeof(CDHCryptLib::SMALL_PRIMES)/sizeof(DWORD);

#ifdef SYSTEM_LITTLE_ENDIAN
const UINT CDHCryptLib::_SHA_MASK_[4]={0x00000000, 0x000000ff, 0x0000ffff, 0x00ffffff};
const UINT CDHCryptLib::_SHA_BITS_[4]={0x00000080, 0x00008000, 0x00800000, 0x80000000};
#else
const UINT CDHCryptLib::_SHA_MASK_[4]={0x00000000, 0xff000000, 0xffff0000, 0xffffff00};
const UINT CDHCryptLib::_SHA_BITS_[4]={0x80000000, 0x00800000, 0x00008000, 0x00000080};
#endif

DWORD CDHCryptLib::BNAdd(DWORD C[], const DWORD A[], const DWORD B[], const UINT nSize)
{	
	DWORD k=0;
	for (UINT i = 0; i < nSize; i++)
	{
		C[i] = A[i] + k;
		if(C[i]>=k)
			k=0;
		else
			k=1;

		C[i] += B[i];
		if (C[i] < B[i])
			k++;	
	}
	return k;
}

UINT CDHCryptLib::BNUiceil(double x)
{
	UINT c;
	if (x < 0) return 0;
	c = (UINT)x;
	if ((x - c) > 0.0)
		c++;
	return c;
}

void CDHCryptLib::SHA1Hash(unsigned char *_pOutDigest, const unsigned char *_pData,UINT nSize)
{
	if ( !_pOutDigest || !_pData )
		return;

	SHA1_STATETYPE csha1;
	memset(&csha1,0,sizeof(csha1));
	SHA1_Start(&csha1);
	SHA1_Hash(_pData,nSize,&csha1);
	SHA1_Finish(_pOutDigest,&csha1);
}


void CDHCryptLib::SHA1_Transform(SHA1_STATETYPE* _pcsha1)
{

	UINT   w[80], i, a, b, c, d, e, t;

	for (i = 0; i < SHA1_BLOCK_SIZE / 4; ++i)
		w[i] = SHA_BLOCK32(_pcsha1->wbuf[i]);

	for (i = SHA1_BLOCK_SIZE / 4; i < 80; ++i)
		w[i] = rotate32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

	a = _pcsha1->hash[0];
	b = _pcsha1->hash[1];
	c = _pcsha1->hash[2];
	d = _pcsha1->hash[3];
	e = _pcsha1->hash[4];

	for(i = 0; i < 20; ++i)
	{
		sha_round(F0to19, 0x5a827999);    
	}

	for(i = 20; i < 40; ++i)
	{
		sha_round(F20to39, 0x6ed9eba1);
	}

	for(i = 40; i < 60; ++i)
	{
		sha_round(F40to59, 0x8f1bbcdc);
	}

	for(i = 60; i < 80; ++i)
	{
		sha_round(F60to79, 0xca62c1d6);
	}

	_pcsha1->hash[0] += a; 
	_pcsha1->hash[1] += b; 
	_pcsha1->hash[2] += c; 
	_pcsha1->hash[3] += d; 
	_pcsha1->hash[4] += e;
}

void CDHCryptLib::SHA1_Start(SHA1_STATETYPE *_pcsha1)
{

	_pcsha1->hash[0] = 0x67452301;
	_pcsha1->hash[1] = 0xefcdab89;
	_pcsha1->hash[2] = 0x98badcfe;
	_pcsha1->hash[3] = 0x10325476;
	_pcsha1->hash[4] = 0xc3d2e1f0;
	_pcsha1->count[0] = 0;
	_pcsha1->count[1] = 0;
}

void CDHCryptLib::SHA1_Finish(unsigned char* _pShaValue, SHA1_STATETYPE* _pcsha1)
{
	UINT    i = (UINT)(_pcsha1->count[0] & (SHA1_BLOCK_SIZE - 1));
	_pcsha1->wbuf[i >> 2] = (_pcsha1->wbuf[i >> 2] & _SHA_MASK_[i & 3]) | _SHA_BITS_[i & 3];

	if(i > SHA1_BLOCK_SIZE - 9)
	{
		if(i < 60) _pcsha1->wbuf[15] = 0;
		SHA1_Transform(_pcsha1);
		i = 0;
	}
	else   
		i = (i >> 2) + 1;

	while(i < 14) 
		_pcsha1->wbuf[i++] = 0;


	_pcsha1->wbuf[14] = SHA_BLOCK32((_pcsha1->count[1] << 3) | (_pcsha1->count[0] >> 29));
	_pcsha1->wbuf[15] = SHA_BLOCK32(_pcsha1->count[0] << 3);

	SHA1_Transform(_pcsha1);


	for(i = 0; i < SHA1_DIGEST_SIZE; ++i)
		_pShaValue[i] = (unsigned char)(_pcsha1->hash[i >> 2] >> 8 * (~i & 3));
}

void CDHCryptLib::SHA1_Hash(const unsigned char *_pData, unsigned int _iSize, SHA1_STATETYPE* _pcsha1)
{
	UINT ipos = (UINT)(_pcsha1->count[0] & (SHA1_BLOCK_SIZE - 1));
	UINT ispace = SHA1_BLOCK_SIZE - ipos;
	unsigned char *pData=(unsigned char *)_pData;
	if((_pcsha1->count[0] += _iSize) < _iSize)
		++(_pcsha1->count[1]);
	while(_iSize >= ispace)     
	{
		memcpy(((unsigned char*)_pcsha1->wbuf) + ipos, pData, ispace);
		ipos = 0; 
		_iSize -= ispace; 
		pData += ispace; 
		ispace = SHA1_BLOCK_SIZE; 
		SHA1_Transform(_pcsha1);
	}	
	memcpy(((unsigned char*)_pcsha1->wbuf) + ipos, pData, _iSize);
}

inline DWORD CDHCryptLib::BNAdddw(DWORD w[], const DWORD u[], DWORD v, UINT nSize)
{
	DWORD k=0;
	w[0] = u[0] + v;
	k=(w[0] >= v) ? 0:1;	
	for (UINT j = 1; j < nSize; j++)
	{
		w[j] = u[j] + k;
		k=(w[j] >= k) ? 0:1;
	}
	return k;	
}

DWORD CDHCryptLib::BNSubtract(DWORD C[], const DWORD A[], const DWORD B[],const UINT nSize)
{
	DWORD  k=0;
	for (UINT i = 0; i < nSize; i++)
	{
		C[i] = A[i] - k;
		if (C[i] > _MAXIMUMNR_ - k)
			k = 1;
		else
			k = 0;
		C[i] -= B[i];
		if (C[i] > _MAXIMUMNR_  - B[i])
			k++;
	}	
	return k;
}

DWORD CDHCryptLib::BNSubtractdw(DWORD w[], const DWORD u[], DWORD v, UINT nSize)
{
	DWORD k=0;
	w[0] = u[0] - v;
	if (w[0] > _MAXIMUMNR_- v)
		k = 1;
	else
		k = 0;
	for (UINT j = 1; j < nSize; j++)
	{
		w[j] = u[j] - k;
		if (w[j] > _MAXIMUMNR_ - k)
			k = 1;
		else
			k = 0;
	}	
	return k;	
}

int CDHCryptLib::BNIsEqual(const DWORD a[], const DWORD b[], UINT nSize)
{
	if ( nSize <= 0 ) 
		return FALSE;

	while ( nSize-- )
	{
		if ( a[nSize] != b[nSize] )
			return FALSE;	
	}	
	return TRUE;
}

void CDHCryptLib::BNSetZero(DWORD A[], UINT nSize)
{
	while ( nSize-- )
		A[nSize] = 0;
}

int CDHCryptLib::BNIsZero(const DWORD a[], UINT nSize)
{
	if (nSize == 0) 
		return FALSE;

	for (UINT i = 0; i < nSize; i++)	
	{
		if (a[i] != 0)
			return FALSE;	
	}

	return TRUE;
}

void CDHCryptLib::BNSetEqual(DWORD a[], const DWORD b[], UINT nSize)
{
	for (UINT i = 0; i < nSize; i++)
	{
		a[i] = b[i];
	}	
}

void CDHCryptLib::BNSetEqualdw(DWORD a[], const DWORD d, UINT nSize)
{
	if(nSize<=0)
		return;
	BNSetZero(a,nSize);
	a[0]=d;
}

int CDHCryptLib::BNCompare(const DWORD a[], const DWORD b[], UINT nSize)
{
	if ( nSize <= 0 ) 
		return 0;
	while (nSize--)
	{
		if (a[nSize] > b[nSize])
			return 1;
		if (a[nSize] < b[nSize])
			return -1;
	}
	return 0;
}

inline int CDHCryptLib::BNComparedw(const DWORD a[], DWORD b, UINT nSize)
{
	if (nSize == 0) return (b ? -1 : 0);

	for (UINT i = 1; i < nSize; i++)
	{
		if (a[i] != 0)
			return 1;
	}

	if ( a[0] < b) 
		return -1;
	else if ( a[0] > b )
		return 1;	

	return 0;
}

inline DWORD CDHCryptLib::BNShiftLeft(DWORD a[], const DWORD *b, UINT x, UINT nSize)
{
	DWORD mask, carry, nextcarry;
	UINT i=0;

	if ( x >= sizeof(DWORD)*8 )
		return 0;

	mask = _HIBITMASK_;
	for (i = 1; i < x; i++)
	{
		mask = (mask >> 1) | mask;
	}
	if (x == 0) mask = 0x0;

	UINT y = (sizeof(DWORD)*8) - x;
	carry = 0;
	for (i = 0; i < nSize; i++)
	{
		nextcarry = (b[i] & mask) >> y;
		a[i] = b[i] << x | carry;
		carry = nextcarry;
	}

	return carry;
}

inline DWORD CDHCryptLib::BNShiftRight(DWORD a[], const DWORD *b, DWORD x, DWORD nSize)
{

	DWORD mask, carry, nextcarry;
	UINT i=0;
	if ( x >= (sizeof(DWORD)*8) )
		return 0;
	mask = 0x1;
	for (i = 1; i < x; i++)
	{
		mask = (mask << 1) | mask;
	}
	if (x == 0) mask = 0x0;

	UINT y = (sizeof(DWORD)*8) - x;
	carry = 0;
	i = nSize;
	while (i--)
	{
		nextcarry = (b[i] & mask) << y;
		a[i] = b[i] >> x | carry;
		carry = nextcarry;
	}

	return carry;	
}

inline DWORD CDHCryptLib::BNDividedw(DWORD q[], const DWORD u[], DWORD v, UINT nSize)
{
	UINT j;
	DWORD t[2], r;
	UINT shift;
	DWORD bitmask, overflow, *uu;

	if (nSize == 0) return 0;
	if (v == 0)	return 0;

	bitmask = _HIBITMASK_;
	for (shift = 0; shift < (sizeof(DWORD)*8); shift++)
	{
		if (v & bitmask)
			break;
		bitmask >>= 1;
	}	
	v <<= shift;
	overflow = BNShiftLeft(q, u, shift, nSize);
	uu = q;
	r = overflow;	
	j = nSize;
	while (j--)
	{
		t[0] = uu[j];
		t[1] = r;
		overflow = BNDivideHelper(&q[j], &r, t, v);
	}
	r >>= shift;	
	return r;
}

DWORD CDHCryptLib::BNMultiply(DWORD C[], const DWORD A[], const DWORD B[], const UINT nSize)
{
	DWORD  k, tmp[2];
	UINT m, n;
	m = n = nSize;
	for ( UINT i = 0; i < 2 * m; i++)
		C[i] = 0;

	for ( UINT j = 0; j < n; j++)
	{
		if (B[j] == 0)
		{
			C[j + m] = 0;
		}
		else
		{
			k = 0;
			for (UINT i = 0; i < m; i++)
			{
				BNMultiplyHelper(tmp, A[i], B[j]);
				tmp[0] += k;
				if (tmp[0] < k)
					tmp[1]++;
				tmp[0] += C[i+j];
				if (tmp[0] < C[i+j])
					tmp[1]++;
				k = tmp[1];
				C[i+j] = tmp[0];

			}	
			C[j+m] = k;
		}
	}	
	return 0;
}

DWORD CDHCryptLib::BNMultiplydw(DWORD w[], const DWORD u[], DWORD v, UINT nSize)
{
	DWORD k, t[2];
	UINT j;
	if (v == 0) 
	{
		for (j = 0; j < nSize; j++)
			w[j] = 0;
		return 0;
	}
	k = 0;
	for (j = 0; j < nSize; j++)
	{
		BNMultiplyHelper(t, u[j], v);
		w[j] = t[0] + k;
		if (w[j] < k)
			t[1]++;
		k = t[1];
	}
	return k;	
}

inline DWORD CDHCryptLib::BNMultSub(DWORD wn, DWORD w[], const DWORD v[], DWORD q, UINT n)
{
	DWORD k, t[2];
	UINT i;

	if ( q == 0 )
		return wn;

	k = 0;

	for (i = 0; i < n; i++)
	{
		BNMultiplyHelper(t, q, v[i]);
		w[i] -= k;
		if (w[i] > _MAXIMUMNR_ - k)
			k = 1;
		else
			k = 0;
		w[i] -= t[0];
		if (w[i] > _MAXIMUMNR_ - t[0])
			k++;
		k += t[1];
	}
	wn -= k;
	return wn;	
}

inline int CDHCryptLib::BNMultiplyHelper(DWORD p[2], const DWORD x, const DWORD y)
{
#ifdef _USEOPTIMIZEASM_
	__asm
	{
		mov eax, x
			xor edx, edx
			mul y
			; Product in edx:eax
			mov ebx, p
			mov dword ptr [ebx], eax
			mov dword ptr [ebx+4], edx
	}
#else 
	DWORD x0, y0, x1, y1;
	DWORD t, u, carry;
	x0 = LOHALF(x);
	x1 = HIHALF(x);
	y0 = LOHALF(y);
	y1 = HIHALF(y);
	p[0] = x0 * y0;
	t = x0 * y1;
	u = x1 * y0;
	t += u;
	if (t < u)
		carry = 1;
	else
		carry = 0;
	carry = TOHIGH(carry) + HIHALF(t);
	t = TOHIGH(t);
	p[0] += t;
	if (p[0] < t)
		carry++;
	p[1] = x1 * y1;
	p[1] += carry;

#endif
	return 0;
}

inline void CDHCryptLib::BNMultSubHelper(DWORD uu[], DWORD qhat, DWORD v1, DWORD v0)
{
	DWORD p0, p1, t;
	p0 = qhat * v0;
	p1 = qhat * v1;
	t = p0 + TOHIGH(LOHALF(p1));
	uu[0] -= t;
	if (uu[0] > _MAXIMUMNR_ - t)
		uu[1]--;
	uu[1] -= HIHALF(p1);

}

inline int CDHCryptLib::BNQhatTooBigHelper(DWORD qhat, DWORD rhat, DWORD vn2, DWORD ujn2)
{
	DWORD t[2];
	BNMultiplyHelper(t, qhat, vn2);
	if ( t[1] < rhat )
		return 0;
	else if ( t[1] > rhat )
		return 1;
	else if ( t[0] > ujn2 )
		return 1;
	return 0;	
}

inline DWORD CDHCryptLib::BNDivideHelper(DWORD *q, DWORD *r, const DWORD u[], DWORD v)
{
	DWORD q2;
	DWORD qhat, rhat, t, v0, v1, u0, u1, u2, u3;
	DWORD uu[2];
	DWORD B= _MAXHALFNR_+1;

	if (!(v & _HIBITMASK_))
	{	
		*q = *r = 0;
		return _MAXIMUMNR_;
	}

	v0 = LOHALF(v);
	v1 = HIHALF(v);
	u0 = LOHALF(u[0]);
	u1 = HIHALF(u[0]);
	u2 = LOHALF(u[1]);
	u3 = HIHALF(u[1]);
	qhat = (u3 < v1 ? 0 : 1);
	if (qhat > 0)
	{
		rhat = u3 - v1;
		t = TOHIGH(rhat) | u2;
		if (v0 > t)
			qhat--;
	}
	uu[0] = u[1];
	uu[1] = 0;		
	if (qhat > 0)
	{

		BNMultSubHelper(uu, qhat, v1, v0);
		if (HIHALF(uu[1]) != 0)
		{
			uu[0] += v;
			uu[1] = 0;
			qhat--;
		}
	}
	q2 = qhat;
	t = uu[0];
	qhat = t / v1;
	rhat = t - qhat * v1;
	t = TOHIGH(rhat) | u1;
	if ( (qhat == B) || (qhat * v0 > t) )
	{
		qhat--;
		rhat += v1;
		t = TOHIGH(rhat) | u1;
		if ((rhat < B) && (qhat * v0 > t))
			qhat--;
	}
	uu[1] = HIHALF(uu[0]);	
	uu[0] = TOHIGH(LOHALF(uu[0])) | u1;	
	BNMultSubHelper(uu, qhat, v1, v0);
	if ( HIHALF(uu[1]) != 0 )
	{	
		qhat--;
		uu[0] += v;
		uu[1] = 0;
	}
	*q = TOHIGH(qhat);
	t = uu[0];
	qhat = t / v1;
	rhat = t - qhat * v1;
	t = TOHIGH(rhat) | u0;
	if ( (qhat == B) || (qhat * v0 > t) )
	{
		qhat--;
		rhat += v1;
		t = TOHIGH(rhat) | u0;
		if ((rhat < B) && (qhat * v0 > t))
			qhat--;
	}
	uu[1] = HIHALF(uu[0]);
	uu[0] = TOHIGH(LOHALF(uu[0])) | u0;	
	BNMultSubHelper(uu, qhat, v1, v0);
	if (HIHALF(uu[1]) != 0)
	{	
		qhat--;
		uu[0] += v;
		uu[1] = 0;
	}
	*q |= LOHALF(qhat);
	*r = uu[0];
	return q2;
}

int CDHCryptLib::BNDivide(DWORD q[], DWORD r[], const DWORD u[], UINT usize, DWORD v[], UINT vsize)
{
	UINT shift;
	int n, m, j;
	DWORD bitmask, overflow;
	DWORD qhat, rhat, t[2];
	DWORD *uu, *ww;
	int qhatOK, cmp;
	BNSetZero(q, usize);
	BNSetZero(r, usize);

	n = (int)BNSizeof(v, vsize);
	m = (int)BNSizeof(u, usize);
	m -= n;

	if ( n == 0 )
		return -1;

	if ( n == 1 )
	{
		r[0] = BNDividedw(q, u, v[0], usize);
		return 0;
	}

	if ( m < 0 )
	{
		BNSetEqual(r, u, usize);
		return 0;
	}

	if ( m == 0 )
	{
		cmp = BNCompare(u, v, (UINT)n);
		if (cmp < 0)
		{
			BNSetEqual(r, u, usize);
			return 0;
		}
		else if (cmp == 0)
		{
			BNSetEqualdw(q, 1, usize);
			return 0;
		}
	}

	bitmask =  _HIBITMASK_;
	for ( shift = 0; shift < 32; shift++ )
	{
		if (v[n-1] & bitmask)
			break;
		bitmask >>= 1;
	}

	overflow = BNShiftLeft(v, v, shift, n);
	overflow = BNShiftLeft(r, u, shift, n + m);
	t[0] = overflow;	
	uu = r;	
	for ( j = m; j >= 0; j-- )
	{
		qhatOK = 0;
		t[1] = t[0];
		t[0] = uu[j+n-1];
		overflow = BNDivideHelper(&qhat, &rhat, t, v[n-1]);
		if ( overflow )
		{	
			rhat = uu[j+n-1];
			rhat += v[n-1];
			qhat = _MAXIMUMNR_;
			if (rhat < v[n-1])	
				qhatOK = 1;
		}
		if (qhat && !qhatOK && BNQhatTooBigHelper(qhat, rhat, v[n-2], uu[j+n-2]))
		{	
			rhat += v[n-1];
			qhat--;
			if (!(rhat < v[n-1]))
				if (BNQhatTooBigHelper(qhat, rhat, v[n-2], uu[j+n-2]))
					qhat--;
		}
		ww = &uu[j];
		overflow = BNMultSub(t[1], ww, v, qhat, (UINT)n);
		q[j] = qhat;
		if (overflow)
		{	
			q[j]--;
			overflow = BNAdd(ww, ww, v, (UINT)n);
		}
		t[0] = uu[j+n-1];	
	}	
	for (j = n; j < m+n; j++)
		uu[j] = 0;
	BNShiftRight(r, r, shift, n);
	BNShiftRight(v, v, shift, n);
	return 0;
}

inline DWORD CDHCryptLib::BNModdw(DWORD a[], DWORD d, UINT nSize)
{
	DWORD *q=NULL;
	DWORD r = 0;
	q = BNAlloc(nSize * 2);
	if(q!=NULL)
	{
		r = BNDividedw(q, a, d, nSize);
		BNFree(&q);
	}
	return r;
}

DWORD CDHCryptLib::BNMod(DWORD r[], const DWORD u[], UINT nUSize, DWORD v[], UINT nVSize)
{
	DWORD *qq, *rr;
	UINT nn = max(nUSize, nVSize);
	qq = BNAlloc(nUSize);
	rr = BNAlloc(nn);
	BNDivide(qq, rr, u, nUSize, v, nVSize);
	BNSetEqual(r, rr, nVSize);
	BNFree(&rr);
	BNFree(&qq);
	return 0;
}

DWORD CDHCryptLib::BNModMult(DWORD a[], const DWORD x[], const DWORD y[], const DWORD m[], UINT nSize)
{
	DWORD *p;
	DWORD *tm;
	p = BNAlloc(nSize * 2);
	tm = BNAlloc(nSize);
	BNSetEqual(tm, m, nSize);
	BNMultiply(p, x, y, nSize);
	BNMod(a, p, nSize * 2, tm, nSize);
	BNFree(&p);
	BNFree(&tm);
	return 0;	
}

UINT CDHCryptLib::BNSizeof(const DWORD A[], UINT nSize)
{
	while ( nSize-- )
	{
		if ( A[nSize] != 0 )
			return (++nSize);
	}
	return 0;
}

UINT CDHCryptLib::BNBitLength(const DWORD *d, UINT nSize)
{
	UINT n, i, bits;
	DWORD mask;

	if ( !d || nSize == 0 )
		return 0;

	n = BNSizeof(d, nSize);
	if (0 == n) return 0;

	DWORD dwLastWord= d[n-1];
	DWORD dwDummY=0;
	mask =_HIBITMASK_;
	for (i = 0; mask > 0;i++)
	{
		if (dwLastWord & mask)
			break;
		mask >>= 1;
	}

	bits = n * (sizeof(DWORD)*8) - i;

	return bits;
}

DWORD * CDHCryptLib::BNAlloc(UINT nSize)
{
	DWORD* p=NULL;
	if(nSize<=0)
		return NULL;

	p=(DWORD*)calloc(nSize, sizeof(DWORD));
	return p; 
}

void CDHCryptLib::BNFree(DWORD **p)
{
	if (*p!=NULL)
	{
		free(*p);
		*p = NULL;
	}
}

int CDHCryptLib::BNModExp(DWORD yout[], const DWORD x[], const DWORD e[], const DWORD m[], UINT nSize)
{
	if ( nSize <= 0 ) 
		return -1;

	DWORD mask;
	UINT n;
	DWORD *t1, *t2, *t3, *tm, *y;

	const UINT nn = nSize * 2;

	t1 = BNAlloc(nn);
	if(t1==NULL)
	{
		return -1;
	}
	t2 = BNAlloc(nn);
	if(t2==NULL)
	{
		BNFree(&t1);
		return -1;
	}
	t3 = BNAlloc(nn);
	if(t3==NULL)
	{
		BNFree(&t1);
		BNFree(&t2);
		return -1;
	}
	tm = BNAlloc(nSize);
	if(tm==NULL)
	{
		BNFree(&t1);
		BNFree(&t2);
		BNFree(&t3);
		return -1;
	}
	y = BNAlloc(nSize);
	if(y==NULL)
	{
		BNFree(&t1);
		BNFree(&t2);
		BNFree(&t3);
		BNFree(&tm);
		return -1;
	}

	BNSetEqual(tm, m, nSize);

	n = BNSizeof(e, nSize);
	for (mask = _HIBITMASK_; mask > 0; mask >>= 1)
	{
		if (e[n-1] & mask)
			break;
	}

	if ( mask==1 )
	{
		mask=_HIBITMASK_;
		n--;
	}else
		mask >>=1; 

	BNSetEqual(y, x, nSize);

	while ( n )
	{

		BNModSquareTmp(y, y, tm, nSize, t1, t2, t3);	
		if (mask & e[n-1])
			BNMultTmp(y, y, x, tm, nSize, t1, t2, t3);

		if ( mask==1 )
		{
			mask=_HIBITMASK_;
			n--;
		}else
			mask >>=1; 
	}

	BNSetEqual(yout, y, nSize);

	BNFree(&t1);
	BNFree(&t2);
	BNFree(&t3);
	BNFree(&tm);
	BNFree(&y);
	return 0;
}

inline int CDHCryptLib::BNMultTmp(DWORD a[], const DWORD x[], const DWORD y[], DWORD m[], UINT nSize, DWORD temp[], DWORD tqq[], DWORD trr[])
{
	BNMultiply(temp, x, y, nSize);
	BNModuloTmp(a, temp, nSize * 2, m, nSize, tqq, trr);
	return 0; 
}

inline int CDHCryptLib::BNModuloTmp(DWORD r[], const DWORD u[], UINT nUSize, DWORD v[], UINT nVSize, DWORD tqq[], DWORD trr[])
{
	BNDivide(tqq, trr, u, nUSize, v, nVSize);
	BNSetEqual(r, trr, nVSize);	
	return 0;
}

inline int CDHCryptLib::BNModSquareTmp(DWORD a[], const DWORD x[], DWORD m[], UINT nSize, DWORD temp[], DWORD tqq[], DWORD trr[])
{
	BNSquare(temp, x, nSize);

	BNModuloTmp(a, temp, nSize * 2, m, nSize, tqq, trr);
	return 0;
}

inline int CDHCryptLib::BNSquare(DWORD w[], const DWORD x[], UINT nSize)
{
	DWORD k, p[2], u[2], cbit, carry;
	UINT i, j, t, i2, cpos;
	t = nSize;

	i2 = t << 1;

	for (i = 0; i < i2; i++)
		w[i] = 0;

	carry = 0;
	cpos = i2-1;

	for (i = 0; i < t; i++)
	{

		i2 = i << 1; 
		BNMultiplyHelper(p, x[i], x[i]);
		p[0] += w[i2];

		if (p[0] < w[i2])
			p[1]++;
		k = 0;	
		if ( i2 == cpos && carry )
		{
			p[1] += carry;
			if (p[1] < carry)
				k++;
			carry = 0;
		}

		u[0] = p[1];
		u[1] = k;
		w[i2] = p[0];

		k = 0;
		for ( j = i+1; j < t; j++ )
		{
			BNMultiplyHelper(p, x[j], x[i]);
			cbit = (p[0] & _HIBITMASK_) != 0;
			k =  (p[1] & _HIBITMASK_) != 0;
			p[0] <<= 1;
			p[1] <<= 1;
			p[1] |= cbit;

			p[0] += u[0];
			if (p[0] < u[0])
			{
				p[1]++;
				if (p[1] == 0)
					k++;
			}
			p[1] += u[1];
			if (p[1] < u[1])
				k++;

			p[0] += w[i+j];
			if (p[0] < w[i+j])
			{
				p[1]++;
				if (p[1] == 0)
					k++;
			}
			if ((i+j) == cpos && carry)
			{
				p[1] += carry;
				if (p[1] < carry)
					k++;
				carry = 0;
			}
			u[0] = p[1];
			u[1] = k;
			w[i+j] = p[0];
		}

		carry = u[1];
		w[i+t] = u[0];
		cpos = i+t;
	}	
	return 0;
}

int CDHCryptLib::BNModInv(DWORD inv[], const DWORD u[], const DWORD v[], UINT nSize)
{
	DWORD *u1, *u3, *v1, *v3, *t1, *t3, *q, *w;
	u1=u3=v1=v3=t1=t3=q=w=NULL;
	int bIterations;
	int result;

	u1 = BNAlloc(nSize);
	if ( u1==NULL )
	{
		return -1;
	}
	u3 = BNAlloc(nSize);
	if ( u3==NULL )
	{
		BNFree(&u1);
		return -1;
	}
	v1 = BNAlloc(nSize);
	if ( v1==NULL )
	{
		BNFree(&u1);
		BNFree(&u3);
		return -1;
	}
	v3 = BNAlloc(nSize);
	if ( v3==NULL )
	{
		BNFree(&u1);
		BNFree(&u3);
		BNFree(&v1);
		return -1;
	}


	t1 = BNAlloc(nSize);

	if ( t1==NULL )
	{
		BNFree(&u1);
		BNFree(&u3);
		BNFree(&v1);
		BNFree(&v3);
		return -1;
	}

	t3 = BNAlloc(nSize);
	if ( t3==NULL )
	{
		BNFree(&u1);
		BNFree(&u3);
		BNFree(&v1);
		BNFree(&v3);
		BNFree(&t1);
		return -1;
	}

	q  = BNAlloc(nSize);
	if ( q==NULL )
	{
		BNFree(&u1);
		BNFree(&u3);
		BNFree(&v1);
		BNFree(&v3);
		BNFree(&t1);
		BNFree(&t3);
		return -1;
	}
	w  = BNAlloc(2 * nSize);
	if ( w==NULL )
	{
		BNFree(&u1);
		BNFree(&u3);
		BNFree(&v1);
		BNFree(&v3);
		BNFree(&t1);
		BNFree(&t3);
		BNFree(&q);
		return -1;
	}

	BNSetEqualdw(u1, 1, nSize);

	BNSetEqual(u3, u, nSize);
	BNSetZero(v1, nSize);
	BNSetEqual(v3, v, nSize);

	bIterations = 1;	
	while ( !BNIsZero(v3, nSize) )		
	{					
		BNDivide(q, t3, u3, nSize, v3, nSize);

		BNMultiply(w, q, v1, nSize);	

		BNAdd(t1, u1, w, nSize);		

		BNSetEqual(u1, v1, nSize);
		BNSetEqual(v1, t1, nSize);
		BNSetEqual(u3, v3, nSize);
		BNSetEqual(v3, t3, nSize);
		bIterations = -bIterations;
	}

	if (bIterations < 0)
		BNSubtract(inv, v, u1, nSize);	
	else
		BNSetEqual(inv, u1, nSize);	


	if (BNComparedw(u3, 1, nSize) != 0)
	{
		result = 1;
		BNSetZero(inv, nSize);
	}
	else
		result = 0;

	BNSetZero(u1, nSize);
	BNSetZero(v1, nSize);
	BNSetZero(t1, nSize);
	BNSetZero(u3, nSize);
	BNSetZero(v3, nSize);
	BNSetZero(t3, nSize);
	BNSetZero(q, nSize);
	BNSetZero(w, 2*nSize);
	BNFree(&u1);
	BNFree(&v1);
	BNFree(&t1);
	BNFree(&u3);
	BNFree(&v3);
	BNFree(&t3);
	BNFree(&q);
	BNFree(&w);
	return 0;

}

int CDHCryptLib::BNGcd(DWORD g[], const DWORD x[], const DWORD y[], UINT nSize)
{
	DWORD *yy, *xx;	
	yy = BNAlloc(nSize);

	if( yy==NULL )
		return -1;

	xx = BNAlloc(nSize);

	if( xx==NULL )
	{
		BNFree(&yy);
		return -1;
	}

	BNSetZero(yy, nSize);
	BNSetZero(xx, nSize);
	BNSetEqual(xx, x, nSize);
	BNSetEqual(yy, y, nSize);
	BNSetEqual(g, yy, nSize);		

	while ( !BNIsZero(xx, nSize) )	
	{
		BNSetEqual(g, xx, nSize);
		BNMod(xx, yy, nSize, xx, nSize);	
		BNSetEqual(yy, g, nSize);	
	}
	BNSetZero(xx, nSize);
	BNSetZero(yy, nSize);
	BNFree(&xx);
	BNFree(&yy);

	return 0;
}

int CDHCryptLib::BNRabinMiller(const DWORD w[], UINT nSize, UINT t)
{

	DWORD *m, *a, *b, *z, *w1, *j;
	DWORD maxrand;
	int failed;
	BOOL bisprime;
	UINT i;

	if (BNComparedw(w, 1, nSize) <= 0) 
		return 0;
	m = BNAlloc(nSize);
	if ( m==NULL ) 
	{
		return FALSE;
	}
	a = BNAlloc(nSize);
	if ( a==NULL )  
	{
		BNFree(&m);
		return FALSE;
	}
	b = BNAlloc(nSize);
	if ( b==NULL ) 
	{
		BNFree(&m);
		BNFree(&a);
		return FALSE;
	}
	z = BNAlloc(nSize);
	if ( z==NULL ) 
	{
		BNFree(&m);
		BNFree(&a);
		BNFree(&b);
		return FALSE;
	}
	w1 = BNAlloc(nSize);
	if ( w1==NULL ) 
	{
		BNFree(&m);
		BNFree(&a);
		BNFree(&b);
		BNFree(&z);
		return FALSE;
	}
	j = BNAlloc(nSize);
	if ( j==NULL ) 
	{
		BNFree(&m);
		BNFree(&a);
		BNFree(&b);
		BNFree(&z);
		BNFree(&w1);
		return FALSE;
	}

	BNSubtractdw(w1, w, 1, nSize);	
	BNSetEqual(m, w1, nSize);		

	for (BNSetZero(a, nSize); (!(m[0]&0x1));  BNAdddw(a, a, 1, nSize))
	{
		BNShiftRight(m, m, 1, nSize);
	}

	if ( BNSizeof(w, nSize) == 1 )
		maxrand = w[0] - 1;
	else
		maxrand = _MAXIMUMNR_;

	bisprime = TRUE;
	for (i = 0; i < t; i++)
	{
		failed = 1;
		BNSetZero(b, nSize);
		do
		{
			b[0] = RandBetween(2, maxrand);
		} while (BNCompare(b, w, nSize) >= 0);

		BNSetZero(j, nSize);
		BNModExp(z, b, m, w, nSize);
		do
		{	
			if ((BNIsZero(j, nSize) && BNComparedw(z, 1, nSize) == 0) || (BNCompare(z, w1, nSize) == 0))
			{
				failed = 0;
				break;
			}

			if ( !BNIsZero(j, nSize) && (BNComparedw(z, 1, nSize) == 0) )
			{
				failed = 1;
				break;
			}

			BNAdddw(j, j, 1, nSize);
			if ( BNCompare(j, a, nSize) < 0 )
				BNModMult(z, z, z, w, nSize);

		} while (BNCompare(j, a, nSize) < 0);

		if ( failed )
		{
			bisprime = FALSE;
			break;
		}
	}	

	BNSetZero(m, nSize);
	BNSetZero(a, nSize);
	BNSetZero(b, nSize);
	BNSetZero(z, nSize);
	BNSetZero(w1, nSize);
	BNSetZero(j, nSize);
	BNFree(&m);
	BNFree(&a);
	BNFree(&b);
	BNFree(&z);
	BNFree(&w1);
	BNFree(&j);	
	return bisprime;
}

BOOL CDHCryptLib::MTInit(BYTE *pRandomPool, UINT nSize)
{
	if ( pRandomPool==NULL || nSize<624*4 ) 
	{
		if ( nSize>0&&pRandomPool )
		{
			memcpy(&m_mtbuffer,pRandomPool,nSize);	  
		}

		if ( nSize<624*4 )
		{
			BYTE *pmtbuffer=(BYTE*)&m_mtbuffer;
			MTCollectEntropy(pmtbuffer+nSize,624*4-nSize);	
		}		

		m_bSeeded=TRUE;
	}
	m_mtIndex=624;
	return m_bSeeded;
}

BOOL CDHCryptLib::MTCollectEntropy(BYTE *pRandomPool, UINT nSize)
{
#ifdef _WINDOWS
	SYSTEMTIME st;
	FILETIME ft;
	MEMORYSTATUS ms;
	SHA1_STATETYPE csha1;
	UINT nCollected = 0; 
	BYTE EntropyBucket[SHA1_DIGEST_SIZE];
	BYTE *pEntropyBucket=(BYTE*)EntropyBucket;
	DWORD dwRes=0;
	DWORD dwTick=0;
	ms.dwLength = sizeof(MEMORYSTATUS);
	memset(&csha1,0,sizeof(csha1));
	SHA1_Start(&csha1);

	while ( nSize-nCollected>0 )
	{	
		SHA1_Hash(pEntropyBucket,SHA1_DIGEST_SIZE,&csha1);
		dwRes=GetCurrentProcessId();
		SHA1_Hash((BYTE*)&dwRes,sizeof(DWORD),&csha1);
		dwRes=GetCurrentThreadId();
		SHA1_Hash((BYTE*)&dwRes,sizeof(DWORD),&csha1);
		GetSystemTime(&st);
		SystemTimeToFileTime(&st, &ft);
		SHA1_Hash((BYTE*)&ft,sizeof(FILETIME),&csha1);
		dwTick = GetTickCount();
		SHA1_Hash((BYTE*)&dwTick,sizeof(DWORD),&csha1);
		GlobalMemoryStatus(&ms);
		SHA1_Hash((BYTE*)&ms, sizeof(MEMORYSTATUS),&csha1);
		SHA1_Finish(EntropyBucket,&csha1);
		if ( nSize-nCollected<SHA1_DIGEST_SIZE )
		{
			memcpy(pRandomPool+nCollected,pEntropyBucket,nSize-nCollected);
			nCollected+=nSize-nCollected;
		}
		else
		{
			memcpy(pRandomPool+nCollected,pEntropyBucket,SHA1_DIGEST_SIZE);
			nCollected+=SHA1_DIGEST_SIZE;
		}
	}
#else
	SHA1_STATETYPE csha1;
    UINT nCollected = 0; 
    BYTE EntropyBucket[SHA1_DIGEST_SIZE];
    BYTE *pEntropyBucket=(BYTE*)EntropyBucket;


    pid_t processId = 0;
    pid_t threadId = 0;
#ifdef __APPLE__
    uint64_t t_spec;
    time_t t_spec_realtime;
    vm_statistics_data_t vmStats;
#else
    struct timespec t_spec;
    struct timespec t_spec_realtime;
    struct sysinfo s_info;
#endif

    memset(&csha1,0,sizeof(csha1));
    SHA1_Start(&csha1);

    while ( nSize-nCollected>0 )
    {	
        SHA1_Hash(pEntropyBucket,SHA1_DIGEST_SIZE,&csha1);
        processId=getpid();
        SHA1_Hash((BYTE*)&processId,sizeof(pid_t),&csha1);
        threadId=gettid();
        SHA1_Hash((BYTE*)&threadId,sizeof(pid_t),&csha1);
#ifdef __APPLE__
        t_spec = mach_absolute_time();
        SHA1_Hash((BYTE*)&t_spec,sizeof(t_spec),&csha1);
        t_spec_realtime = time(NULL);
        SHA1_Hash((BYTE*)&t_spec_realtime,sizeof(t_spec_realtime),&csha1);
        
        mach_msg_type_number_t infoCount = HOST_VM_INFO_COUNT;
        kern_return_t kernReturn = host_statistics(mach_host_self(),
                                                   HOST_VM_INFO,
                                                   (host_info_t)&vmStats,
                                                   &infoCount);
        SHA1_Hash((BYTE*)&vmStats, sizeof( vmStats),&csha1);
#else
        clock_gettime(CLOCK_MONOTONIC, &t_spec);
        SHA1_Hash((BYTE*)&t_spec,sizeof(struct timespec),&csha1);
        clock_gettime(CLOCK_REALTIME, &t_spec_realtime);
        SHA1_Hash((BYTE*)&t_spec_realtime,sizeof(struct timespec),&csha1);
#ifdef ANDROID
        sysinfo(s_info);
#else
		sysinfo(&s_info);
#endif
        SHA1_Hash((BYTE*)&s_info, sizeof(struct sysinfo),&csha1);
#endif
        SHA1_Finish(EntropyBucket,&csha1);
        if ( nSize-nCollected<SHA1_DIGEST_SIZE )
        {
            memcpy(pRandomPool+nCollected,pEntropyBucket,nSize-nCollected);
            nCollected+=nSize-nCollected;
        }
        else
        {
            memcpy(pRandomPool+nCollected,pEntropyBucket,SHA1_DIGEST_SIZE);
            nCollected+=SHA1_DIGEST_SIZE;
        }
    }
#endif //_WINDOWS

	return TRUE; 
}

DWORD CDHCryptLib::MTRandom()
{
	if( !m_bSeeded )
		MTInit();

	if ( m_mtIndex >= 624 )
	{
		m_mtIndex = 0;
		int i = 0;
		int s;
		for (; i < 624 - 397; i++) {
			s = (m_mtbuffer[i] & 0x80000000) | (m_mtbuffer[i+1] & 0x7FFFFFFF);
			m_mtbuffer[i] = m_mtbuffer[i + 397] ^ (s >> 1) ^ ((s & 1) * 0x9908B0DF);
		}
		for (; i < 623; i++) {
			s = (m_mtbuffer[i] & 0x80000000) | (m_mtbuffer[i+1] & 0x7FFFFFFF);
			m_mtbuffer[i] = m_mtbuffer[i - (624 - 397)] ^ (s >> 1) ^ ((s & 1) * 0x9908B0DF);
		}

		s = (m_mtbuffer[623] & 0x80000000) | (m_mtbuffer[0] & 0x7FFFFFFF);
		m_mtbuffer[623] = m_mtbuffer[396] ^ (s >> 1) ^ ((s & 1) * 0x9908B0DF);
	}
	DWORD tmp=m_mtbuffer[m_mtIndex++];
	tmp  ^= (tmp >> 11);
	tmp ^= (tmp << 7) & 0x9D2C5680UL;
	tmp ^= (tmp << 15) & 0xEFC60000UL;
	return tmp ^ (tmp >> 18);
}

inline DWORD CDHCryptLib::RandBetween(DWORD dwLower, DWORD dwUpper)
{

	DWORD d, range;
	unsigned char *bp;
	int i, nbits;
	DWORD mask;

	if ( dwUpper <= dwLower ) 
	{
		return dwLower;
	}
	range = dwUpper - dwLower;

	do
	{
		bp = (unsigned char *)&d;
		for (i = 0; i < sizeof(DWORD); i++)
		{
			bp[i] = BYTE(MTRandom() & 0xFF);
		}


		mask = _HIBITMASK_;
		for (nbits = sizeof(DWORD)*8; nbits > 0; nbits--, mask >>= 1)
		{
			if (range & mask)
				break;
		}
		if (nbits < sizeof(DWORD)*8)
		{
			mask <<= 1;
			mask--;
		}
		else
			mask = _MAXIMUMNR_;

		d &= mask;

	} while (d > range); 

	return (dwLower + d);
}

int CDHCryptLib::BNIsPrime(DWORD W[], UINT nSize, UINT nrRounds)
{
	if ((!(W[0] & 0x1)))
		return 0;

	if (BNComparedw(W, SMALL_PRIMES[_NUMBEROFPRIMES_-1], nSize) > 0)
	{
		for (UINT i = 0; i < _NUMBEROFPRIMES_; i++)
		{
			if (BNModdw(W, SMALL_PRIMES[i], nSize) == 0)
				return FALSE;
		}
	}
	else
	{	
		for (UINT i = 0; i < _NUMBEROFPRIMES_; i++)
		{
			if (BNComparedw(W, SMALL_PRIMES[i], nSize) == 0)
				return TRUE;
		}
		return FALSE;
	}
	return BNRabinMiller(W, nSize, nrRounds);
}

int CDHCryptLib::BNMakePrime(DWORD p[], UINT nSize, PBYTE pEntropyPool, UINT nSizeEntropyPool)
{
	if ( pEntropyPool )
	{
		MTInit(pEntropyPool,nSizeEntropyPool);
	}

	for ( UINT i = 0; i < nSize; i++ )
		p[i] = MTRandom();	

	p[nSize - 1] |= _HIBITMASK_;
	p[0] |= 0x1;

	while (!BNIsPrime(p, nSize, 64) )
	{
		BNAdddw(p, p, 2, nSize);

		if (!(p[nSize - 1] & _HIBITMASK_))
			return -1;	
	}
	return BNBitLength(p,nSize);
}

UINT CDHCryptLib::BNMakeRandomNr(DWORD a[], UINT nSize)
{
	UINT  i, n, bits;
	DWORD mask;

	n = (UINT)RandBetween(1, nSize);
	for ( i = 0; i < n; i++) 
		a[i] = MTRandom();
	for ( i = n; i < nSize; i++)
		a[i] = 0;
	bits = (UINT)RandBetween(0, 16*sizeof(DWORD));

	if ( bits != 0 && bits < 8*sizeof(DWORD) )
	{	
		mask = _HIBITMASK_;
		for (i = 1; i < bits; i++)
		{
			mask |= (mask >> 1);
		}
		mask = ~mask;
		a[n-1] &= mask;
	}
	return n;	
}

