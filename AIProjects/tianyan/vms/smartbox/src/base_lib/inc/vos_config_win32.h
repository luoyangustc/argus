#ifndef __VOS_COMPAT_OS_WIN32_H__
#define __VOS_COMPAT_OS_WIN32_H__

#define VOS_OS_NAME		    "win32"

#define WIN32_LEAN_AND_MEAN

#ifndef _MSC_VER
#  error "This header file is only for Visual C compiler!"
#endif

#define VOS_CC_NAME	    "msvc"
#define VOS_CC_VER_1	    (_MSC_VER/100)
#define VOS_CC_VER_2	    (_MSC_VER%100)
#define VOS_CC_VER_3	    0

/**
*	Windows   XP                _WIN32_WINNT>=0x0501    
*	Windows   2000            _WIN32_WINNT>=0x0500    
*	Windows   NT   4.0        _WIN32_WINNT>=0x0400    
*	Windows   Me                _WIN32_WINDOWS=0x0490    
*	Windows   98                _WIN32_WINDOWS>=0x0410
*  从 Visual C++ 2008 开始，Visual C++ 不支持面向 Windows 95、Windows 98、Windows ME 或 Windows NT
**/
#define VOS_WIN32_WINNT		    0x0501 //0x0400
#ifndef _WIN32_WINNT
#  define _WIN32_WINNT		    VOS_WIN32_WINNT
#endif

#define VOS_HAS_WINSOCK2_H			1
#define VOS_HAS_HIGH_RES_TIMER		0

#ifndef VOS_OS_HAS_CHECK_STACK
#define VOS_OS_HAS_CHECK_STACK	0
#endif


/* Disable CRT deprecation warnings. */
#if VOS_CC_VER_1 >= 8 && !defined(_CRT_SECURE_NO_DEPRECATE)
#   define _CRT_SECURE_NO_DEPRECATE
#endif
#if VOS_CC_VER_1 >= 8 && !defined(_CRT_SECURE_NO_WARNINGS)
#   define _CRT_SECURE_NO_WARNINGS
    /* The above doesn't seem to work, at least on VS2005, so lets use
     * this construct as well.
     */
#   pragma warning(disable: 4996)
#endif

#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4611) // not wise to mix setjmp with C++
#pragma warning(disable: 4514) // unref. inline function has been removed
#ifdef NDEBUG
#  pragma warning(disable: 4702) // unreachable code
#  pragma warning(disable: 4710) // function is not inlined.
#  pragma warning(disable: 4711) // function selected for auto inline expansion
#endif

#ifdef UNICODE
#   define _UNICODE    1
#else
#   define _UNICODE    0
#endif

#define ATOMIC_VALUE_TYPE		long


#ifndef VOS_HAS_SEMAPHORE
#  define VOS_HAS_SEMAPHORE	            1
#endif

#ifndef VOS_HAS_EVENT_OBJ
#  define VOS_HAS_EVENT_OBJ	            1
#endif

#ifndef VOS_HAS_CTYPE_H
#   define VOS_HAS_CTYPE_H      1
#endif

#ifndef VOS_HAS_ASSERT_H
#   define VOS_HAS_ASSERT_H      1
#endif

#ifndef VOS_ENABLE_EXTRA_CHECK
#   define VOS_ENABLE_EXTRA_CHECK    1
#endif

#define VOS_HAS_INT64	                1

#define ATOMIC_VALUE_TYPE   long

#ifdef __cplusplus
#  define VOS_INLINE_SPECIFIER	__inline  //inline
#else
#  define VOS_INLINE_SPECIFIER	static __inline
#endif


#endif	/* __VOS_COMPAT_OS_WIN32_H__ */


