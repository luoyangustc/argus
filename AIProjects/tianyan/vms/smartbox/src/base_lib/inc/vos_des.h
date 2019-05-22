#ifndef __VOS_UTIL_DES_H__
#define __VOS_UTIL_DES_H__

#include "vos_types.h"

#undef  EXT
#ifndef __UTIL_DES_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

typedef char ElemType;


EXT int DES_Encrypt(unsigned char *pPlainData, int iPlainDataLength, 
				unsigned char *pKeyStr, int iKeyLength, 
				unsigned char *pCipherData, int iCipherMaxSize,
				int*   piCipherSize);  
EXT int DES_Decrypt(unsigned char *pCipherData, int iCipherDataLength, 
				unsigned char *pKeyStr, int iKeyLength, 
				unsigned char *pPlainData, int iPlainMaxSize,
				int*   piPlainSize); 

VOS_END_DECL

#endif		//__VOS_UTIL_DES_H__

