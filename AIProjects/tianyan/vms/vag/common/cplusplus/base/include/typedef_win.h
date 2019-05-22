#ifndef __TYPE_DEF_WIN_H__
#define __TYPE_DEF_WIN_H__

#ifdef _WINDOWS
#define INT64_ARG   "%I64d"
#define UINT64_ARG  "%I64u"
#else
#define INT64_ARG   "%lld"
#define UINT64_ARG  "%llu"
#endif // _WINDOWS

#ifdef _WINDOWS
#define THREAD_HANDLE_TYPE HANDLE
#define THREAD_HANDLE_INIT_VALUE 0
#define THREAD_RETURN_TYPE unsigned __stdcall
#define THREAD_RETURN_VALUE  0
#else
#define THREAD_HANDLE_TYPE pthread_t
#define THREAD_HANDLE_INIT_VALUE 0
#define THREAD_RETURN_TYPE void*
#define THREAD_RETURN_VALUE  (void*)(0)
#endif // _WINDOWS

#ifdef _WINDOWS
#include <winsock2.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif//_WINDOWS

#define INOUT

#ifndef _WINDOWS
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <wchar.h>

#define FILE_BEGIN              0
#define FILE_CURRENT            1
#define FILE_END                2


#ifndef NULL
#define NULL    ((void *)0)
#endif

#ifndef FALSE
#define FALSE               0
#endif

#ifndef TRUE
#define TRUE                1
#endif

#ifndef INFINITE
#define INFINITE 0xffffffffu
#endif

#define	IN
#define	OUT

#define CALLBACK
#define WINAPI
#define WINAPIV
#define APIENTRY    	    WINAPI
#define APIPRIVATE
#define PASCAL

#define FAR
#define NEAR
#define CONST		        const
#define CDECL

#define VOID                void

//// algorithm identifier definitions
#define ALG_TYPE_ANY                    (0)
#define ALG_SID_MD5                     3
#define ALG_SID_SHA1                    4
#define ALG_CLASS_ANY                   (0)
#define ALG_CLASS_SIGNATURE             (1 << 13)
#define ALG_CLASS_MSG_ENCRYPT           (2 << 13)
#define ALG_CLASS_DATA_ENCRYPT          (3 << 13)
#define ALG_CLASS_HASH                  (4 << 13)
#define ALG_CLASS_KEY_EXCHANGE          (5 << 13)
#define ALG_CLASS_ALL                   (7 << 13)

#define CALG_MD5                (ALG_CLASS_HASH | ALG_TYPE_ANY | ALG_SID_MD5)
#define CALG_SHA1               (ALG_CLASS_HASH | ALG_TYPE_ANY | ALG_SID_SHA1)

typedef void * LPVOID;
typedef unsigned int       DWORD;
typedef unsigned char       BYTE;
typedef BYTE*		    PBYTE;
typedef unsigned short      WORD;
typedef unsigned long       ULONG_PTR;
typedef ULONG_PTR           SIZE_T;


typedef signed char         INT8, *PINT8;
typedef signed short        INT16, *PINT16;
typedef signed int          INT32, *PINT32;
typedef unsigned char       UINT8, *PUINT8;
typedef unsigned short      UINT16, *PUINT16;
typedef unsigned int        UINT32, *PUINT32;

typedef long long           __int64;
typedef long long           INT64;
typedef unsigned long long  ULONGLONG;
typedef long long	    LONGLONG;

typedef void                *PVOID;
typedef char                CHAR;
typedef short               SHORT;
typedef long                LONG;
typedef SHORT               *PSHORT;  
typedef LONG                *PLONG;    

typedef unsigned char 		UCHAR;
typedef unsigned short 		USHORT;
typedef unsigned long		ULONG;
typedef unsigned int		UINT;


#ifdef __APPLE__
#if __LP64__ || (TARGET_OS_EMBEDDED && !TARGET_OS_IPHONE) || TARGET_OS_WIN32 || NS_BUILD_32_LIKE_64
typedef bool		        BOOL;
#else
typedef signed char		        BOOL;
#endif
#else
typedef int		        BOOL;
#endif //__APPLE__
typedef ULONG *		    PULONG;
typedef USHORT *	    PUSHORT;
typedef UCHAR *		    PUCHAR;
typedef char *		    PSZ;
typedef int               	INT;
typedef unsigned int       	*PUINT;

//typedef unsigned short      WCHAR;
typedef wchar_t             WCHAR;
typedef WCHAR               *PWCHAR;
typedef WCHAR               *LPWCH, *PWCH;
typedef CONST WCHAR         *LPCWCH, *PCWCH;
typedef WCHAR               *NWPSTR;
typedef WCHAR               *LPWSTR, *PWSTR;
typedef CONST WCHAR         *LPCWSTR, *PCWSTR;

typedef CHAR                *PCHAR;
typedef CHAR                *LPCH, *PCH;
typedef CONST CHAR          *LPCCH, *PCCH;
typedef CHAR                *NPSTR;
typedef CHAR                *LPSTR, *PSTR;
typedef CONST CHAR          *LPCSTR, *PCSTR;
typedef char                TCHAR, *PTCHAR;
typedef unsigned char       TBYTE , *PTBYTE ;
typedef LPSTR               LPTCH, PTCH;
typedef LPSTR               PTSTR, LPTSTR;
typedef LPCSTR              LPCTSTR;

typedef int (FAR WINAPI *FARPROC)();
typedef int (NEAR WINAPI *NEARPROC)();
typedef int (WINAPI *PROC)();

typedef UINT WPARAM;
typedef LONG LPARAM;
typedef LONG LRESULT;
typedef LONG HRESULT;

typedef DWORD   COLORREF;
typedef DWORD   *LPCOLORREF;

typedef PVOID HANDLE;

typedef HANDLE *PHANDLE;
typedef HANDLE NEAR         *SPHANDLE;
typedef HANDLE FAR          *LPHANDLE;
typedef HANDLE              HGLOBAL;
typedef HANDLE              HLOCAL;
typedef HANDLE              GLOBALHANDLE;
typedef HANDLE              LOCALHANDLE;

typedef WORD                ATOM;

typedef struct hwnd *	    HWND;
typedef struct hdc *	    HDC;
typedef struct hcursor      *HCURSOR;
typedef struct hgdiobj      *HGDIOBJ;
typedef struct hgdiobj      *HBRUSH;
typedef struct hgdiobj      *HPEN;
typedef struct hgdiobj      *HFONT;
typedef struct hgdiobj      *HBITMAP;
typedef struct hgdiobj      *HRGN;
typedef struct hgdiobj      *HPALETTE;
typedef HANDLE		        HICON;
typedef HANDLE		        HINSTANCE;
typedef HANDLE		        HMENU;
typedef HANDLE	            HKEY;
typedef	WORD	            INTERNET_PORT;

#ifndef S_OK
#define S_OK ((HRESULT)0x00000000L)
#endif
#ifndef S_FALSE
#define S_FALSE ((HRESULT)0x00000001L)
#endif
#ifndef	E_FAIL
#define E_FAIL ((HRESULT)0x80000008L)
#endif
#ifndef	NOERROR
#define	NOERROR	0
#endif

#ifndef	INVALID_HANDLE_VALUE
#define	INVALID_HANDLE_VALUE	((int)-1)
#endif

typedef	int	SOCKET;

typedef	struct sockaddr	SOCKADDR;
typedef	struct sockaddr *PSOCKADDR;
typedef	struct sockaddr_in	SOCKADDR_IN;
typedef	struct hostent	HOSTENT;
typedef	struct linger	LINGER;

typedef	struct	_WSABUF	{
	ULONG	   len;     /* the length of the buffer */
	char FAR * buf;     /* the pointer to the buffer */
} WSABUF, FAR * LPWSABUF;

#ifndef	INVALID_SOCKET
#define	INVALID_SOCKET	((int)-1)
#endif

#define SOCKET_ERROR -1

typedef ULONG_PTR	HCRYPTPROV;
typedef ULONG_PTR 	HCRYPTKEY;
typedef ULONG_PTR 	HCRYPTHASH;
typedef	unsigned int	ALG_ID;
#ifndef __64BITS__
typedef	unsigned long long	UINT64;
#else
typedef	unsigned long	UINT64;
#endif

typedef int INT_PTR;

#ifndef ULONG_MAX
#define ULONG_MAX 4294967295u
#endif

#define STDMETHOD(x) virtual HRESULT x

#ifndef PURE
#define PURE =0
#endif

#ifndef __stdcall
#define __stdcall
#endif
#define STDMETHODCALLTYPE       __stdcall
#define STDMETHODIMP            HRESULT STDMETHODCALLTYPE
#define STDAPICALLTYPE          __stdcall

#define HMODULE                 HINSTANCE

typedef union _LARGE_INTEGER {
    struct {
        DWORD LowPart;
        LONG HighPart;
    };
    struct {
        DWORD LowPart;
        LONG HighPart;
    } u;
    LONGLONG QuadPart;
} LARGE_INTEGER;

#define MAX_PATH (512)

#define PATH_DELIMITER ("/")
#define PATH_DELIMITER_C ('/')

#define sprintf_s snprintf

#define lstrlen strlen

#define CTRY	try{
#define CCATCH	}						\
	catch(...){}

#define _T 

#endif

#endif //__TYPE_DEF_WIN_H__

