#include "TokenCrypt.h"
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#undef min
#undef max
#define min(x,y)	 ((x)<(y)?(x):(y))
#define max(x,y)	((x)>(y)?(x):(y))

CTokenCrypt::CTokenCrypt()
{
	m_mtIndex=0;
	m_bSeeded=FALSE;
}

CTokenCrypt::~CTokenCrypt()
{

}

int CTokenCrypt::BNMakeRSAPrime(DWORD p[], DWORD ee, UINT nSize,UINT nMaximumRetry)
{
	UINT nRet=-1; 
	for ( UINT i=0; i<nMaximumRetry; i++ )
	{
		nRet=BNMakePrime(p,nSize);
		if(nRet>0&&BNModdw(p, ee,nSize)!=1)
			break;
	}
	return nRet;
}

int CTokenCrypt::RSAGenerateKey(DWORD n[], DWORD d[], DWORD p[], DWORD q[], DWORD dP[], DWORD dQ[], DWORD qInv[], UINT nSize, UINT nPSize,UINT nQSize,DWORD e, BYTE* pSeedData,UINT nSeedData)
{
	if( nSize<max(nPSize,nQSize)*2 )
		return -30;

	UINT nPrimeSize=max(nPSize,nQSize);
	UINT nNSize=0; 
	UINT nDSize=0;
	DWORD *pG=BNAlloc(nSize);
	if ( pG==NULL )
	{
		return -1;
	}

	DWORD *pP1=BNAlloc(nSize);
	if ( pP1==NULL )
	{
		BNFree(&pG);
		return -2;
	}

	DWORD *pQ1=BNAlloc(nSize);
	if ( pQ1==NULL )
	{
		BNFree(&pG);
		BNFree(&pP1);
		return -3;
	}

	DWORD *pPhi=BNAlloc(nSize);
	if ( pPhi==NULL )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		return -4;
	}

	DWORD *pE=BNAlloc(nSize);
	if ( pE==NULL )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		return -5;
	}	

	if ( pSeedData!=NULL && nSeedData>0 ) 
		MTInit(pSeedData,nSeedData/2);
	else 
		MTInit();

	int nRet=BNMakeRSAPrime(p,e,nPSize);
	if ( nRet<=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -6;
	}

	if ( pSeedData!=NULL && nSeedData>0)  
		MTInit(pSeedData+nSeedData/2,nSeedData/2);
	else 
		MTInit();

	nRet=BNMakeRSAPrime(q,e,nQSize);
	if ( nRet<=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -7;
	}

	if ( BNIsEqual(p,q,nPrimeSize) )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -8;
	}

	BNSetEqualdw(pE,e,nSize);
	if ( BNCompare(p, q,nPrimeSize) < 1 )
	{	
		BNSetEqual(pG, p,nPrimeSize);
		BNSetEqual(p, q,nPrimeSize);
		BNSetEqual(q, pG,nPrimeSize);
	}

	if ( BNSubtractdw(pP1,p,1,nPrimeSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -8;
	}

	if ( BNSubtractdw(pQ1,q,1,nPrimeSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -8;
	}

	BNGcd(pG, pP1, pE,nPrimeSize);

	if ( BNComparedw(pG,1,nPrimeSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -9;
	}
	
	BNGcd(pG, pQ1, pE,nPrimeSize);

	if ( BNComparedw(pG,1,nPrimeSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -9;
	}
	BNMultiply(n, p, q,nPrimeSize);

	nNSize=BNSizeof(n,nSize);

	if ( BNIsZero(n,nNSize)) 
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -11;
	}

	BNMultiply(pPhi, pP1, pQ1,nPrimeSize);

	if ( BNIsZero(pPhi,nSize))  
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -11;
	}

	nRet = BNModInv(d, pE, pPhi,nSize);

	nDSize=BNSizeof(d,nSize);

	if ( BNIsZero(d,nDSize) || nRet!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -11;
	}

	BNSetZero(pG,nSize);
	BNModMult(pG, pE, d, pPhi,max(nDSize,nPrimeSize*2));
	
	if ( BNComparedw(pG,1,nSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -11;
	}

	if ( BNModInv(dP, pE, pP1,nPrimeSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -12;
	}

	if ( BNModInv(dQ, pE, pQ1,nPrimeSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -13;
	}

	if ( BNModInv(qInv, q, p,nSize)!=0 )
	{
		BNFree(&pG);
		BNFree(&pP1);
		BNFree(&pQ1);
		BNFree(&pPhi);
		BNFree(&pE);
		return -14;
	}

	BNFree(&pG);
	BNFree(&pP1);
	BNFree(&pQ1);
	BNFree(&pPhi);
	BNFree(&pE);
	return nNSize;	
}

int CTokenCrypt::RSAGenerateKey(DWORD n[], DWORD d[], UINT nSize, DWORD e /*= 65537*/, BYTE* pSeedData/*=NULL*/, UINT nSeedData/*=0*/)
{
	UINT nPSize = nSize / 2;
	UINT nQSize = nSize - nPSize;
	if( nSize<max(nPSize,nQSize)*2 )
	{
		return -30;
	}

	UINT nPrimeSize=max(nPSize,nQSize);
	UINT nNSize=0; 
	UINT nDSize=0;

	scoped_array<DWORD> pG(new DWORD[nSize]);
	scoped_array<DWORD> pP1(new DWORD[nSize]);
	scoped_array<DWORD> pQ1(new DWORD[nSize]);
	scoped_array<DWORD> pPhi(new DWORD[nSize]);
	scoped_array<DWORD> pE(new DWORD[nSize]);
	scoped_array<DWORD> p(new DWORD[nSize]);
	scoped_array<DWORD> q(new DWORD[nSize]);
	
	if(NULL==pG.get() || NULL==pP1.get() || NULL==pQ1.get() || NULL==pPhi.get() || NULL==pE.get() || NULL==p.get() || NULL==q.get())
	{
		return -1;
	}

	do 
	{
		memset(pG.get(), 0, nSize*sizeof(DWORD));
		memset(pP1.get(), 0, nSize*sizeof(DWORD));
		memset(pQ1.get(), 0, nSize*sizeof(DWORD));
		memset(pPhi.get(), 0, nSize*sizeof(DWORD));
		memset(pE.get(), 0, nSize*sizeof(DWORD));
		memset(p.get(), 0, nSize*sizeof(DWORD));
		memset(q.get(), 0, nSize*sizeof(DWORD));
	} while(false);

	if ( pSeedData!=NULL && nSeedData>0 )
	{
		MTInit(pSeedData,nSeedData/2);
	}
	else
	{
		MTInit();
	}

	int nRet=BNMakeRSAPrime(p.get(), e, nPSize);
	if ( nRet<=0 )
	{
		return -2;
	}

	if ( pSeedData!=NULL && nSeedData>0) 
	{
		MTInit(pSeedData+nSeedData/2,nSeedData/2);
	}
	else
	{
		MTInit();
	}

	nRet=BNMakeRSAPrime(q.get(), e, nQSize);
	if ( nRet<=0 )
	{
		return -3;
	}
	if ( BNIsEqual(p.get(), q.get(), nPrimeSize) )
	{
		return -4;
	}
	
	BNSetEqualdw(pE.get(), e, nSize);

	if ( BNCompare(p.get(), q.get(), nPrimeSize) < 1 )
	{	
		BNSetEqual(pG.get(), p.get(), nPrimeSize);
		BNSetEqual(p.get(), q.get(), nPrimeSize);
		BNSetEqual(q.get(), pG.get(), nPrimeSize);
	}

	if ( BNSubtractdw(pP1.get() , p.get(),1,nPrimeSize)!=0 )
	{
		return -5;
	}
	
	if ( BNSubtractdw(pQ1.get(), q.get(), 1,nPrimeSize)!=0 )
	{
		return -6;
	}

	BNGcd(pG.get(), pP1.get(), pE.get(), nPrimeSize);

	if ( BNComparedw(pG.get(),1,nPrimeSize)!=0 )
	{
		return -7;
	}

	BNGcd(pG.get(), pQ1.get(), pE.get(), nPrimeSize);

	if ( BNComparedw(pG.get(),1,nPrimeSize)!=0 )
	{
		return -8;
	}

	BNMultiply(n, p.get(), q.get(),nPrimeSize);

	nNSize=BNSizeof(n,nSize);

	if ( BNIsZero(n,nNSize)) 
	{
		return -9;
	}

	BNMultiply(pPhi.get(), pP1.get(), pQ1.get(), nPrimeSize);

	if ( BNIsZero(pPhi.get(), nSize))  
	{
		return -10;
	}

	nRet = BNModInv(d, pE.get(), pPhi.get(), nSize);

	nDSize=BNSizeof(d,nSize);


	if ( BNIsZero(d,nDSize) || nRet!=0 )
	{
		return -11;
	}

	BNSetZero(pG.get(), nSize);
	BNModMult(pG.get(), pE.get(), d, pPhi.get(), max(nDSize,nPrimeSize*2));
	
	if ( BNComparedw(pG.get(),1,nSize)!=0 )
	{
		return -12;
	}

	return nNSize;	
}

int CTokenCrypt::RSAEncrypt(DWORD c[], DWORD m[], DWORD n[], UINT nSize, DWORD e)
{
	if ( !c||!m||!n||nSize<=0 )
		return 0;

	int iRet=0;
	DWORD *pE=BNAlloc(nSize);
	if ( pE==NULL )
	{
		return -1;
	}
	BNSetEqualdw(pE,e,nSize);
	iRet=RSAEncrypt(c,m,n,pE,nSize);

	if( pE )
		BNFree(&pE);
	return iRet;
}

int CTokenCrypt::RSAEncrypt(DWORD c[], DWORD m[], DWORD n[], DWORD e[], UINT nSize)
{
	if ( !c || !m || !n || !e || nSize<=0 )
		return -1;
	return  BNModExp(c, m, e, n,nSize);
}

int CTokenCrypt::RSADecryptCRT(DWORD m[],DWORD c[],DWORD p[], DWORD q[], DWORD dP[], DWORD dQ[], DWORD qInv[], UINT nSize)
{
	DWORD dwOverFlow=0;
	DWORD *pm2=BNAlloc(nSize);
	if ( pm2==NULL )
	{
		return -1;
	}
	DWORD *ph=BNAlloc(nSize);

	if ( ph==NULL )
	{
		BNFree(&pm2);
		return -2;
	}

	DWORD *phq=BNAlloc(nSize);

	if ( phq==NULL )
	{
		BNFree(&pm2);
		BNFree(&ph);
		return -3;
	}

	BNSetZero(pm2,nSize);
	BNSetZero(ph,nSize);
	BNSetZero(phq,nSize);

	dwOverFlow+=(DWORD)BNModExp(m, c, dP, p,nSize);
	dwOverFlow+=(DWORD)BNModExp(pm2, c, dQ, q,nSize);

	if ( BNCompare(m, pm2,nSize) < 0 )
	{
		dwOverFlow+=BNAdd(m, m, p,nSize);
	}
	dwOverFlow+=BNSubtract(m, m, pm2,nSize);	
	dwOverFlow+=BNModMult(ph, qInv, m, p,nSize/2);
	dwOverFlow+=BNMultiply(phq, ph, q,nSize/2);
	dwOverFlow+=BNAdd(m, pm2, phq,nSize);

	BNFree(&pm2);
	BNFree(&ph);
	BNFree(&phq);
	return (int)dwOverFlow;
}
