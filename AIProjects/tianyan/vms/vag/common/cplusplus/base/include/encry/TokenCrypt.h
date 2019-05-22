#ifndef __TOKEN_CRYPT_H__
#define __TOKEN_CRYPT_H__

#include "DHCryptLib.h"
#include "../typedef_win.h"
#include <string>
using namespace std;

#include <boost/smart_ptr.hpp>
using namespace boost;
#include<string>
#include<sstream>
using namespace std;

class CTokenCrypt : public CDHCryptLib  
{
public:
	CTokenCrypt();
	virtual ~CTokenCrypt();
	int RSADecryptCRT(DWORD m[],DWORD c[],DWORD p[], DWORD q[], DWORD dP[], DWORD dQ[], DWORD qInv[], UINT nSize);
	int RSAEncrypt(DWORD c[], DWORD m[], DWORD n[], DWORD e[], UINT nSize);
	int RSAEncrypt(DWORD c[],DWORD m[],DWORD n[],UINT nSize,DWORD e);
	int RSAGenerateKey(DWORD n[], DWORD d[], DWORD p[], DWORD q[], DWORD dP[], DWORD dQ[], DWORD qInv[], UINT nSize,UINT nPSize,UINT nQSize,DWORD e=65537, BYTE* pSeedData=NULL,UINT nSeedData=0);
	int RSAGenerateKey(DWORD n[], DWORD d[], UINT nSize, DWORD e = 65537, BYTE* pSeedData=NULL, UINT nSeedData=0);
	int BNMakeRSAPrime(DWORD p[],DWORD ee, UINT nSize,UINT nMaximumRetry=30);
};

#endif // __TOKEN_CRYPT_H__
