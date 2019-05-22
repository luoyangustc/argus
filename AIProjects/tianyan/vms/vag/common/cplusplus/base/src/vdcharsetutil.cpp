#include "vdcharsetutil.h"
//#include <iconv.h>
#ifdef ANDROID
#include "mbs_wcs.h"
#define __LIVE_UTF8_STRING__ "UTF-8"
#else
#define __LIVE_UTF8_STRING__ "UTF8"
#endif
#include <boost/format.hpp>
using namespace boost;


int CCharsetConvertUtil::WCS2UTF8(const wchar_t * a_szSrc,int a_nSrcSize,char* a_szDest,int a_nDestSize)
{
#ifdef _WINDOWS
	return WideCharToMultiByte(CP_UTF8,0,a_szSrc,-1,a_szDest,a_nDestSize,NULL,NULL);
#else
	size_t result;
	iconv_t env;
	env = iconv_open(__LIVE_UTF8_STRING__,"WCHAR_T");

	if( env == (iconv_t)-1 )
	{
	    printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("icon_open WCHAR_T->UTF-8 error:%s %d\n",strerror(errno),errno);
		return -1;
	}

	size_t a_nSrcSize_st = a_nSrcSize;
	size_t a_nDestSize_st = a_nDestSize;

	result = iconv(env,
		(char**)&a_szSrc,
		(size_t*)&a_nSrcSize_st,
		(char**)&a_szDest,
		(size_t*)&a_nDestSize_st		
		);

	if( result == (size_t)-1 )
	{
	    printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
	    printf("iconv WCHAR_T->UTF8 error:%s %d\n",strerror(errno),errno);
	}
	
	iconv_close(env);
	return (int)result;
#endif
}

int CCharsetConvertUtil::WCS2UTF8(const wchar_t * a_szSrc,string& out_string)
{
    size_t stwcsLen = wcslen(a_szSrc);

    char pchar[4096];
    memset(pchar,0,4096);
    
    int iDestLen = 4096;
    int iRet = WCS2UTF8(a_szSrc,stwcsLen*4+4,pchar,iDestLen);
    out_string = pchar;
    return iRet;
}

int CCharsetConvertUtil::MBS2UTF8(const char * a_szSrc,string& out_string)
{
#ifdef ANDROID
	setlocale(LC_ALL,"zh_CN.UTF-8");
#endif
    out_string = a_szSrc;
   // return 1; //to explain 这句是本来就有的,这是不是错了？？？

    size_t stSrcLen = strlen(a_szSrc);
    wchar_t * pwcharBuf = new wchar_t[stSrcLen*4+4];
    memset( pwcharBuf,0,sizeof(wchar_t) * (stSrcLen*4+4) );
#ifdef _WINDOWS
    ::MultiByteToWideChar(CP_ACP,0,a_szSrc,-1,pwcharBuf,stSrcLen*2+2  );
#else
#ifdef ANDROID
    __mbstowcs__(pwcharBuf,a_szSrc,stSrcLen*4+4);
#else
	mbstowcs(pwcharBuf,a_szSrc,stSrcLen*4+4);
#endif
#endif

    int iRet = WCS2UTF8(pwcharBuf,out_string);
    delete []pwcharBuf;
    return iRet;
}

int CCharsetConvertUtil::MBS2UTF16(const char * a_szSrc,char * a_szDest,int a_nDestSize)
{
#ifdef ANDROID
	setlocale(LC_ALL,"zh_CN.UTF-8");
#endif
    size_t stSrcLen = strlen(a_szSrc);
    wchar_t * pwcharBuf = new wchar_t[stSrcLen*2+2];
    memset( pwcharBuf,0,sizeof(wchar_t) * (stSrcLen*2+2) );
#ifdef _WINDOWS
	::MultiByteToWideChar(CP_ACP,0,a_szSrc,-1,pwcharBuf,stSrcLen*2+2  );
	int iRet = WCS2UTF16(pwcharBuf,stSrcLen*2+2,a_szDest,a_nDestSize);
#else

#ifdef ANDROID
	__mbstowcs__(pwcharBuf,a_szSrc,stSrcLen*4+4);
#else
	mbstowcs(pwcharBuf,a_szSrc,stSrcLen*4+4);
#endif

    int iSrcSize = wcslen(pwcharBuf);
    
    int iRet = WCS2UTF16(pwcharBuf,stSrcLen*4+4,a_szDest,a_nDestSize);
#endif
    delete []pwcharBuf;
    return iRet;


}

int CCharsetConvertUtil::UTF82WCS(const char * a_szSrc,wchar_t * a_szDest,int a_nDestSize)
{
#ifdef _WINDOWS
    return MultiByteToWideChar(CP_UTF8,0,a_szSrc,-1,a_szDest,a_nDestSize);
#else
	size_t result;
	iconv_t env;
	size_t size = strlen(a_szSrc)+1;
	env = iconv_open("WCHAR_T",__LIVE_UTF8_STRING__);

	if( env == (iconv_t)-1 )
	{
		printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("icon_open UTF-8->WCHAR_T error:%s %d\n",strerror(errno),errno);
		return -1;
	}

	size_t 	a_nDestSize_st = a_nDestSize;
	result = iconv(env,
		(char**)&a_szSrc,
		(size_t*)&size,
		(char**)&a_szDest,
		(size_t*)&a_nDestSize_st		
		);

	if( result == (size_t)-1 )
	{
	    printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
	    printf("iconv UTF8->WCHAR_T error:%s %d\n",strerror(errno),errno);
	    iconv_close(env);
	    return -1;	
	}
	
	iconv_close(env);
	return (int)result;
#endif
}

int CCharsetConvertUtil::WCS2UTF16(const wchar_t * a_szSrc,int a_nSrcSize,char* a_szDest,int a_nDestSize)
{
#ifdef _WINDOWS
    //memcpy_s( (wchar_t*)a_szDest,a_nDestSize,a_szSrc,a_nSrcSize);
	wcscpy((wchar_t*)a_szDest,a_szSrc);
    return a_nDestSize;
#else
	size_t result;
	iconv_t env;
	//env = iconv_open("UCS-2-INTERNAL","UCS-4-INTERNAL");
	env = iconv_open("UCS-2","WCHAR_T");
	if( env == (iconv_t)-1 )
	{
	    printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("icon_open WCHAR_T->UCS-2 error:%s %d\n",strerror(errno),errno);
		return -1;
	}

	size_t a_nSrcSize_st = a_nSrcSize;
	size_t a_nDestSize_st = a_nDestSize;

	result = iconv(env,
		(char**)&a_szSrc,
		(size_t*)&a_nSrcSize_st,
		(char**)&a_szDest,
		(size_t*)&a_nDestSize_st		
		);

	if( result == (size_t)-1 )
	{
	    printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
	    printf("iconv WCHAR_T->UCS-2 error:%s %d\n",strerror(errno),errno);
	    iconv_close(env);
	    return -1;	
	}
	
	iconv_close(env);
	return (int)result;
#endif
}

int CCharsetConvertUtil::UTF162WCS(const char * a_szSrc,int a_nSrcSize,wchar_t * a_szDest,int a_nDestSize)
{
#ifdef _WINDOWS
    //memcpy_s( a_szDest,a_nDestSize,(const wchar_t*)a_szSrc,a_nSrcSize);

	wcscpy(a_szDest,(wchar_t*)a_szSrc);
    return a_nDestSize;
#else
	size_t result;
	iconv_t env;
	env = iconv_open("WCHAR_T","UCS-2");
	if( env == (iconv_t)-1 )
	{
		printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("icon_open UCS-2->WCHAR_T error:%s %d\n",strerror(errno),errno);
		return -1;
	}

	size_t a_nSrcSize_st = a_nSrcSize;
	size_t a_nDestSize_st = a_nDestSize;

	result = iconv(env,
		(char**)&a_szSrc,
		(size_t*)&a_nSrcSize_st,
		(char**)&a_szDest,
		(size_t*)&a_nDestSize_st		
		);

	if( result == (size_t)-1 )
	{
	    printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
	    printf("iconv UCS-2->WCHAR_T error:%s %d\n",strerror(errno),errno);
	    iconv_close(env);
	    return -1;	
	}
	
	iconv_close(env);
	return (int)result;
#endif
}

int CCharsetConvertUtil::UTF162MBS(const char * a_szSrc,string& out_string)
{
#ifdef ANDROID
	setlocale(LC_ALL,"zh_CN.UTF-8");
#endif
    int a_nSrcSize = 0;
    {
	WORD * pTmp = (WORD*)a_szSrc;
	for(;;pTmp++)
	{
	    WORD wTmp = *pTmp;
	    if( wTmp == 0 )
	    {
		break;
	    }
	    else
	    {
		++a_nSrcSize;
	    }
	}
    }

    ++a_nSrcSize;
    a_nSrcSize *= 2;


    wchar_t a_szDest[4096] = {0};
    int a_nDestSize = 4096;

    int iRet = UTF162WCS(a_szSrc,a_nSrcSize,a_szDest,a_nDestSize);

    size_t stwcsLen = wcslen(a_szDest);
    char* pchar = new char[stwcsLen*2+1];
    memset(pchar,0,stwcsLen*2+1);
#ifdef _WINDOWS
    WideCharToMultiByte( CP_ACP, 0, a_szDest, -1, pchar,stwcsLen*2,NULL,NULL); 
#else
#ifdef ANDROID
    __wcstombs__(pchar,a_szDest,stwcsLen*2);
#else
	 wcstombs(pchar,a_szDest,stwcsLen*2);
#endif
#endif

    out_string = pchar;
    delete []pchar;
    return out_string.size();
}

void CCharsetConvertUtil::MBS2EncodeURL(const string& sURL,OUT string& sEncodeURL)
{
	sEncodeURL = sURL;

	{
		bool bNeedEncode = false;
		string::size_type nLen = sURL.length();
		string::const_iterator it = sURL.begin();
		for(; it != sURL.end(); ++it )
		{
			if( ((*it) & 0x80) || ((*it) == 0x20 ) )
			{
				bNeedEncode = true;
				break;
			}
		}

		if( bNeedEncode == false )
		{
			return;
		}
	} 


	{
		char szBuf[2048];
		memset(szBuf,0,2048);

		int nCurr = 0;
		string::const_iterator it = sURL.begin();
		for(; it != sURL.end();  )
		{
			char c = (*it);
			// 			if( c == '_' )
			// 			{
			// 				string sTmp(sURL.begin(),it);
			// 				szBuf[nCurr++] = c;	
			// 				++it;
			// 			}
			// 			else 
			if( c == 0x2B )//+
			{
				szBuf[nCurr++] = c;	
				++it;
			}
			else if( c == 0x26)//&
			{
				szBuf[nCurr++] = c;	
				++it;
			}
			else if( c == 0x20 )//空格
			{
				char szTmp[] = "%20";
				int nLenTmp = 3;
				memcpy(szBuf+nCurr,szTmp,6);
				nCurr+= 3;
				++it;
			}
			else if( c>0x20 && c< 0x7f )
			{
				szBuf[nCurr++] = c;
				++it;
			}
			else if(c<0x20 && c>0)
			{
				szBuf[nCurr++] = c;
				++it;
			}
			/*
			else if( c < 0x20 && c > 0 )//不知怎么办
			{
				szBuf[nCurr++] = c;
				++it;
			}				  */
			else
			{
				char szTmp[1024];
				memset(szTmp,0,1024);
				int iCurrTmp = 0;

				int nCntTmp = 0;
				do 
				{
					char cTmp = *it;
					if( nCntTmp % 2 == 0 )
					{
						if( cTmp < 0x7f && cTmp > 0)
						{
							break;
						}
					}

					szTmp[iCurrTmp++] = cTmp;

					++it;

					++nCntTmp;
				} while (it != sURL.end());

				szTmp[iCurrTmp] = 0;

				char utf8Buffer[4096];
			
#ifdef _WINDOWS
				//将szTmp转换为UTF8
				wchar_t utf16Buffer[4096];
				int l_Count = 0;
				l_Count = ::MultiByteToWideChar( CP_ACP, 0,szTmp, -1, utf16Buffer, 4096 );
				utf16Buffer[l_Count] = 0;
				l_Count = ::WideCharToMultiByte( CP_UTF8, 0, utf16Buffer, -1, utf8Buffer, 4096, NULL, NULL );
				utf8Buffer[l_Count] = 0;
#else
				string sUTF8String;
				//printf("%s\n",szTmp);
				//MBS2UTF8(szTmp,sUTF8String);
				sUTF8String = szTmp;
				//printf("%s\n",sUTF8String.c_str());

				strcpy(utf8Buffer,sUTF8String.c_str());
				int l_Count = 0;
				l_Count = sUTF8String.length() + 1;
#endif
				boost::format fm("%%%2X");
				for( int i = 0; i< (l_Count-1); ++i )
				{
					fm.clear();					
					fm % ((int)(utf8Buffer[i]) & 0xff );
					memcpy(szBuf+nCurr,fm.str().c_str(),3);
					nCurr += 3;
				}			
			}			
		}

		szBuf [nCurr] = 0;

		sEncodeURL = szBuf;
	}
}

int CCharsetConvertUtil::GB180302UTF8(const char * a_szSrc,string& out_string)
{
	size_t result;
#ifdef _WINDOWS	  
	wchar_t utf16Buffer[4096];
	char utf8Buffer[4096];
	int l_Count = 0;
	l_Count = ::MultiByteToWideChar( CP_ACP, 0,a_szSrc/* sql.c_str()*/, -1, utf16Buffer, 4096 );
	utf16Buffer[l_Count] = 0;
	l_Count = ::WideCharToMultiByte( CP_UTF8, 0, utf16Buffer, -1, utf8Buffer, 4096, NULL, NULL );
	utf8Buffer[l_Count] = 0;

	out_string = utf8Buffer;
	return l_Count;	
#else
	iconv_t env;
	env = iconv_open(__LIVE_UTF8_STRING__,"GB18030");
	if( env == (iconv_t)-1 )
	{
		//LOGD5("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		//LOGD5("icon_open GB18030->UTF-8 error:%s %d\n",strerror(errno),errno);
		return -1;
	}

	size_t a_nSrcSize = strlen(a_szSrc) + 1;
	char szDest[4096] ;
	memset(szDest,0,4096);
	size_t a_nDestSize = 4096;
	
	char * a_szDest = szDest;

	result = iconv(env,
		(char**)&a_szSrc,
		(size_t*)&a_nSrcSize,
		(char**)&a_szDest,
		(size_t*)&a_nDestSize		
		);

	if( result == (size_t)-1 )
	{
		printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("iconv GB18030->UTF8 error:%s %d\n",strerror(errno),errno);
	}

	iconv_close(env);

	out_string = szDest;
#endif
	return (int)result;
}


int CCharsetConvertUtil::UTF82GB18030(const char * a_szSrc,string& out_string)
{
	size_t result;
#ifdef _WINDOWS
	wchar_t utf16Buffer[4096];
	char ansiBuffer[4096];
	int l_Count = 0;
	l_Count = ::MultiByteToWideChar( CP_UTF8, 0, a_szSrc, -1, utf16Buffer, 4096 );
	utf16Buffer[l_Count] = 0;
	l_Count = ::WideCharToMultiByte( CP_ACP, 0, utf16Buffer, -1, ansiBuffer, 4096, NULL, NULL );
	ansiBuffer[l_Count] = 0;
	out_string = ansiBuffer;
	return l_Count;
#else
	iconv_t env;

	env = iconv_open("GB18030",__LIVE_UTF8_STRING__);

	if( env == (iconv_t)-1 )
	{
		printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("icon_open UTF-8->GB18030 error:%s %d\n",strerror(errno),errno);
		return -1;
	}

	size_t a_nSrcSize = strlen(a_szSrc) + 1;
	char szDest[4096] ;
	memset(szDest,0,4096);
	size_t a_nDestSize = 4096;

	char * a_szDest = szDest;

	result = iconv(env,
		(char**)&a_szSrc,
		(size_t*)&a_nSrcSize,
		(char**)&a_szDest,
		(size_t*)&a_nDestSize		
		);

	if( result == (size_t)-1 )
	{
		printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("iconv UTF-8->GB18030 error:%s %d\n",strerror(errno),errno);
	}

	iconv_close(env);

	out_string = szDest;
 #endif
	return (int)result;
}

int CCharsetConvertUtil::UTF82GB18030(const char * a_szSrc,char * a_szDest,int a_nDestSize)
{
	size_t result;
#ifdef _WINDOWS
	wchar_t utf16Buffer[4096];
	char ansiBuffer[4096];
	int l_Count = 0;
	l_Count = ::MultiByteToWideChar( CP_UTF8, 0, a_szSrc, -1, utf16Buffer, 4096 );
	utf16Buffer[l_Count] = 0;
	l_Count = ::WideCharToMultiByte( CP_ACP, 0, utf16Buffer, -1, ansiBuffer, 4096, NULL, NULL );
	ansiBuffer[l_Count] = 0;

	if (a_nDestSize > l_Count)
	{
		strncpy(a_szDest,ansiBuffer,l_Count);
		return l_Count;
	}			   
	return -1;
#else
	iconv_t env;
	env = iconv_open("GB18030",__LIVE_UTF8_STRING__);

	if( env == (iconv_t)-1 )
	{
		printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("icon_open UTF-8->GB18030 error:%s %d\n",strerror(errno),errno);
		return -1;
	}

	size_t a_nSrcSize = strlen(a_szSrc) + 1;
	memset(a_szDest,0,a_nDestSize);

	size_t a_nDestSize_st = a_nDestSize;

	result = iconv(env,
		(char**)&a_szSrc,
		(size_t*)&a_nSrcSize,
		(char**)&a_szDest,
		(size_t*)&a_nDestSize_st		
		);

	if( result == (size_t)-1 )
	{
		printf("%s(%d)-%s:",__FILE__,__LINE__,__FUNCTION__);
		printf("iconv UTF-8->GB18030 error:%s %d\n",strerror(errno),errno);
	}

	iconv_close(env);
 #endif
	return (int)result;
}


string CCharsetConvertUtil::UnicodeToAnsi( const wchar_t* uniStr, int len, UINT CodePage )
{
		string strAnsi;
#ifdef _WINDOWS
#else
        if(!uniStr)
        {
            return "";
        }
		iconv_t cv = iconv_open("GBK", "UCS-4LE");
		if(cv == (iconv_t)-1)
        {
            return "";
        }

        size_t unicode_length = (len + 1) * sizeof(wchar_t);
        char * unicode_string = const_cast<char*>(reinterpret_cast<const char*>(uniStr));

        size_t gbk_length  = unicode_length/2 + 1;
        char * gbk_string = new char[gbk_length];
        char * gbk_org_string = gbk_string;
        memset(gbk_string, 0, gbk_length);

        iconv(cv, &unicode_string, &unicode_length, &gbk_string, &gbk_length);

        iconv_close(cv);

        strAnsi = gbk_org_string;
        delete []gbk_org_string;
  #endif
		return strAnsi;
}

wstring CCharsetConvertUtil::AnsiToUnicode( const char* ansiStr, int len, UINT CodePage )
{
		wstring strUCS;
#ifdef _WINDOWS
#else
        if(!ansiStr)
        {
            return L"";
        }
		iconv_t cv = iconv_open("UCS-4LE", "GBK");
		if(cv == (iconv_t)-1)
        {
            return L"";
        }

        size_t gbk_length = len;
        char * gbk_string = const_cast<char*>(ansiStr);

        size_t unicode_length  = (gbk_length + 1) * sizeof(wchar_t);
        char * unicode_string = new char[unicode_length];
        char * unicode_org_string = unicode_string;
        memset(unicode_string, 0, unicode_length);

        iconv(cv, &gbk_string, &gbk_length, &unicode_string, &unicode_length);

        iconv_close(cv);

        strUCS = reinterpret_cast<wchar_t*>(unicode_org_string);
        delete []unicode_org_string;
   #endif
		return strUCS;

}

