#ifndef __VD_CHARSET_UTIL_H__
#define __VD_CHARSET_UTIL_H__

#ifdef _WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <ctype.h>
#else
#include <wctype.h>
#include <wchar.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iconv.h>
#include "typedef_win.h"
#endif

#include <string>
using namespace std;

class CCharsetConvertUtil
{
public:
    static int WCS2UTF8(const wchar_t * a_szSrc,int a_nSrcSize,char* a_szDest,int a_nDestSize);
    static int WCS2UTF8(const wchar_t * a_szSrc,string& out_string);
    static int MBS2UTF8(const char * a_szSrc,string& out_string);

    static int MBS2UTF16(const char * a_szSrc,char * a_szDest,int a_nDestSize);

    static int UTF162MBS(const char * a_szSrc,string& out_string);

	static int GB180302UTF8(const char * a_szSrc,string& out_string);
	static int UTF82GB18030(const char * a_szSrc,string& out_string);
	static int UTF82GB18030(const char * a_szSrc,char * a_szDest,int a_nDestSize);

    static int UTF82WCS(const char * a_szSrc,wchar_t * a_szDest,int a_nDestSize);
    static int WCS2UTF16(const wchar_t * a_szSrc,int a_nSrcSize,char* a_szDest,int a_nDestSize);
    static int UTF162WCS(const char * a_szSrc,int a_nSrcSize,wchar_t * a_szDest,int a_nDestSize);
	static void MBS2EncodeURL(const string& sURL,OUT string& sEncodeURL);
	static string UnicodeToAnsi( const wchar_t* uniStr, int len, UINT CodePage );
	static wstring AnsiToUnicode( const char* ansiStr, int len, UINT CodePage );
	
};



#endif

